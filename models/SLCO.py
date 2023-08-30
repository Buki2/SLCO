import torch
import torch.nn.functional as F
from torch import nn

from util import box_ops
from util.misc import (NestedTensor, get_world_size, is_dist_avail_and_initialized)

from .backbone import build_backbone

from .transformer import build_visual_encoder
from .decoder import build_vg_decoder
from pytorch_pretrained_bert.modeling import BertModel

from torch.nn.init import kaiming_normal_, kaiming_uniform_
from collections import OrderedDict


class SLCO(nn.Module):
    def __init__(self, pretrained_weights, args=None):
        super().__init__()

        # Image feature encoder (CNN + Transformer encoder)
        self.backbone = build_backbone(args)
        self.trans_encoder = build_visual_encoder(args)
        self.input_proj = nn.Conv2d(self.backbone.num_channels, self.trans_encoder.d_model, kernel_size=1)

        # Text feature encoder (BERT)
        self.bert = BertModel.from_pretrained(args.bert_model)
        self.bert_proj = nn.Linear(args.bert_output_dim, args.hidden_dim)
        self.bert_output_layers = args.bert_output_layers
        for v in self.bert.pooler.parameters():
            v.requires_grad_(False)

        # Cross-modal decoder in category-based grounding
        self.trans_decoder = build_vg_decoder(args)

        hidden_dim = self.trans_encoder.d_model
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)

        # Segment detection
        self.ground_text = GroundTextBlock_multihop(NFilm=3, textdim=self.trans_encoder.d_model,
                                                                visudim=self.trans_encoder.d_model,
                                                                emb_size=self.trans_encoder.d_model,
                                                                fusion='prod', intmd=False)
        # Category association in category-based grounding
        self.mapping_fact1 = MLP(256 * 3, 256 * 2, 256, 2)
        self.mapping_fact2 = MLP(256 * 3, 256 * 2, 256, 2)
        self.mapping_fact3 = MLP(256 * 3, 256 * 2, 256, 2)

    def load_pretrained_weights(self, weights_path):
        def load_weights(module, prefix, weights):
            module_keys = module.state_dict().keys()
            weights_keys = [k for k in weights.keys() if prefix in k]
            update_weights = dict()
            for k in module_keys:
                prefix_k = prefix+'.'+k
                if prefix_k in weights_keys:
                    update_weights[k] = weights[prefix_k]
                else:
                    print(f"Weights of {k} are not pre-loaded.")
            module.load_state_dict(update_weights, strict=False)

        weights = torch.load(weights_path, map_location='cpu')['model']
        load_weights(self.backbone, prefix='backbone', weights=weights)
        load_weights(self.trans_encoder, prefix='transformer', weights=weights)
        load_weights(self.input_proj, prefix='input_proj', weights=weights)

    def forward(self, image, image_mask, word_id, word_mask, knowledge_category_word_id, knowledge_category_word_mask):
        N = image.size(0)

        # Image features
        features, pos = self.backbone(NestedTensor(image, image_mask))
        src, mask = features[-1].decompose()
        assert mask is not None
        img_feat_proj = self.input_proj(src)
        img_feat, mask, pos_embed = self.trans_encoder(img_feat_proj, mask, pos[-1])

        # Text features
        word_feat, _ = self.bert(word_id, token_type_ids=None, attention_mask=word_mask)
        word_feat = torch.stack(word_feat[-self.bert_output_layers:], 1).mean(1)
        word_feat_proj = self.bert_proj(word_feat)

        # Segment detection module
        lang_visu_score, lang_know_score, subsentence_attn_list, _ = self.ground_text(img_feat_proj, word_feat_proj, fsent=None, word_mask=word_mask)
        visual_segment = lang_visu_score.unsqueeze(2) * word_feat_proj + word_feat_proj

        # Category-based grounding module
        # Textual features for knowledge categories
        knowledge_category_word_id1 = knowledge_category_word_id[:, 0, :].squeeze(1)
        knowledge_category_word_mask1 = knowledge_category_word_mask[:, 0, :].squeeze(1)
        knowledge_category_word_id2 = knowledge_category_word_id[:, 1, :].squeeze(1)
        knowledge_category_word_mask2 = knowledge_category_word_mask[:, 1, :].squeeze(1)
        knowledge_category_word_id3 = knowledge_category_word_id[:, 2, :].squeeze(1)
        knowledge_category_word_mask3 = knowledge_category_word_mask[:, 2, :].squeeze(1)
        knowledge_category_feat, _ = self.bert(knowledge_category_word_id1, token_type_ids=None, attention_mask=knowledge_category_word_mask1)
        knowledge_category_feat = torch.stack(knowledge_category_feat[-self.bert_output_layers:], 1).mean(1)
        knowledge_category_feat_proj = self.bert_proj(knowledge_category_feat)
        knowledge_category_feat_proj1 = knowledge_category_feat_proj[:, 1:-1, :]
        knowledge_category_word_mask1 = knowledge_category_word_mask1[:, 1:-1]
        knowledge_category_feat, _ = self.bert(knowledge_category_word_id2, token_type_ids=None, attention_mask=knowledge_category_word_mask2)
        knowledge_category_feat = torch.stack(knowledge_category_feat[-self.bert_output_layers:], 1).mean(1)
        knowledge_category_feat_proj = self.bert_proj(knowledge_category_feat)
        knowledge_category_feat_proj2 = knowledge_category_feat_proj[:, 1:-1, :]
        knowledge_category_feat, _ = self.bert(knowledge_category_word_id3, token_type_ids=None, attention_mask=knowledge_category_word_mask3)
        knowledge_category_feat = torch.stack(knowledge_category_feat[-self.bert_output_layers:], 1).mean(1)
        knowledge_category_feat_proj = self.bert_proj(knowledge_category_feat)
        knowledge_category_feat_proj3 = knowledge_category_feat_proj[:, 1:-1, :]

        # Integrate knowledge categories into visual segments
        cate_asso_beg1 = torch.cat([visual_segment[:, :1, :], knowledge_category_feat_proj1[:, :, :], visual_segment[:, 1:, :]], dim=1)
        cate_asso_beg2 = torch.cat([visual_segment[:, :1, :], knowledge_category_feat_proj2[:, :, :], visual_segment[:, 1:, :]], dim=1)
        cate_asso_beg3 = torch.cat([visual_segment[:, :1, :], knowledge_category_feat_proj3[:, :, :], visual_segment[:, 1:, :]], dim=1)
        cate_asso_end1 = cate_asso_end2 = cate_asso_end3 = cate_asso_mid1 = cate_asso_mid2 = cate_asso_mid3 = torch.zeros_like(cate_asso_beg1)
        for ii in range(N):
            border_end = len(torch.nonzero(word_mask[ii])) - 1
            cate_asso_end1[ii] = torch.cat([visual_segment[ii][:border_end, :], knowledge_category_feat_proj1[ii][:, :], visual_segment[ii][border_end:, :]], dim=0)
            cate_asso_end2[ii] = torch.cat([visual_segment[ii][:border_end, :], knowledge_category_feat_proj2[ii][:, :], visual_segment[ii][border_end:, :]], dim=0)
            cate_asso_end3[ii] = torch.cat([visual_segment[ii][:border_end, :], knowledge_category_feat_proj3[ii][:, :], visual_segment[ii][border_end:, :]], dim=0)
            border_mid = len(torch.nonzero(word_mask[ii])) // 2
            cate_asso_mid1[ii] = torch.cat([visual_segment[ii][:border_mid, :], knowledge_category_feat_proj1[ii][:, :], visual_segment[ii][border_mid:, :]], dim=0)
            cate_asso_mid2[ii] = torch.cat([visual_segment[ii][:border_mid, :], knowledge_category_feat_proj2[ii][:, :], visual_segment[ii][border_mid:, :]], dim=0)
            cate_asso_mid3[ii] = torch.cat([visual_segment[ii][:border_mid, :], knowledge_category_feat_proj3[ii][:, :], visual_segment[ii][border_mid:, :]], dim=0)

        # Category association
        cate_asso_sent1 = self.mapping_fact1(torch.cat([cate_asso_beg1, cate_asso_end1, cate_asso_mid1], dim=-1)).permute(1, 0, 2)
        cate_asso_sent2 = self.mapping_fact2(torch.cat([cate_asso_beg2, cate_asso_end2, cate_asso_mid2], dim=-1)).permute(1, 0, 2)
        cate_asso_sent3 = self.mapping_fact3(torch.cat([cate_asso_beg3, cate_asso_end3, cate_asso_mid3], dim=-1)).permute(1, 0, 2)

        # Transformer decoder
        word_mask = torch.cat([knowledge_category_word_mask1, word_mask], dim=1)
        word_mask = ~word_mask
        results = self.trans_decoder(img_feat, mask, pos_embed, cate_asso_sent1, cate_asso_sent2, cate_asso_sent3, word_mask)
        outputs_coord = self.bbox_embed(results).sigmoid()
        out = {'pred_boxes': outputs_coord[-1]}

        if self.training:
            out['aux_outputs'] = [{'pred_boxes': b} for b in outputs_coord[:-1]]
        return out, subsentence_attn_list, lang_know_score


class VGCriterion(nn.Module):
    def __init__(self, weight_dict, loss_loc, box_xyxy):
        super().__init__()
        self.weight_dict = weight_dict

        self.box_xyxy = box_xyxy

        self.loss_map = {'loss_boxes': self.loss_boxes}

        self.loss_loc = self.loss_map[loss_loc]

    def loss_boxes(self, outputs, target_boxes, num_pos):
        assert 'pred_boxes' in outputs
        src_boxes = outputs['pred_boxes'] # [B, #query, 4]
        target_boxes = target_boxes[:, None].expand_as(src_boxes)

        src_boxes = src_boxes.reshape(-1, 4) # [B*#query, 4]
        target_boxes = target_boxes.reshape(-1, 4) #[B*#query, 4]

        losses = {}
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        losses['l1'] = loss_bbox.sum() / num_pos

        if not self.box_xyxy:
            src_boxes = box_ops.box_cxcywh_to_xyxy(src_boxes)
            target_boxes = box_ops.box_cxcywh_to_xyxy(target_boxes)
        loss_giou = 1 - box_ops.box_pair_giou(src_boxes, target_boxes)
        losses['giou'] = (loss_giou[:, None]).sum() / num_pos
        return losses


    def forward(self, outputs, targets):
        gt_boxes = targets['bbox']
        pred_boxes = outputs['pred_boxes']

        losses = {}
        B, Q, _ = pred_boxes.shape
        num_pos = avg_across_gpus(pred_boxes.new_tensor(B*Q))
        loss = self.loss_loc(outputs, gt_boxes, num_pos)
        losses.update(loss)

        # Apply the loss function to the outputs from all the stages
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                l_dict = self.loss_loc(aux_outputs, gt_boxes, num_pos)
                l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                losses.update(l_dict)

        return losses


class PostProcess(nn.Module):
    def __init__(self, box_xyxy=False):
        super().__init__()
        self.bbox_xyxy = box_xyxy

    @torch.no_grad()
    def forward(self, outputs, target_dict):
        rsz_sizes, ratios, orig_sizes = \
            target_dict['size'], target_dict['ratio'], target_dict['orig_size']
        dxdy = None if 'dxdy' not in target_dict else target_dict['dxdy']

        boxes = outputs['pred_boxes']

        assert len(boxes) == len(rsz_sizes)
        assert rsz_sizes.shape[1] == 2

        boxes = boxes.squeeze(1)

        # Convert to absolute coordinates in the original image
        if not self.bbox_xyxy:
            boxes = box_ops.box_cxcywh_to_xyxy(boxes)
        img_h, img_w = rsz_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct
        if dxdy is not None:
            boxes = boxes - torch.cat([dxdy, dxdy], dim=1)
        boxes = boxes.clamp(min=0)
        ratio_h, ratio_w = ratios.unbind(1)
        boxes = boxes / torch.stack([ratio_w, ratio_h, ratio_w, ratio_h], dim=1)
        if orig_sizes is not None:
            orig_h, orig_w = orig_sizes.unbind(1)
            boxes = torch.min(boxes, torch.stack([orig_w, orig_h, orig_w, orig_h], dim=1))

        return boxes


def avg_across_gpus(v, min=1):
    if is_dist_avail_and_initialized():
        torch.distributed.all_reduce(v)
    return torch.clamp(v.float() / get_world_size(), min=min).item()


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build_vgmodel(args):
    device = torch.device(args.device)

    model = SLCO(pretrained_weights=args.load_weights_path, args=args)

    weight_dict = {'loss_cls': 1, 'l1': args.bbox_loss_coef}
    weight_dict['giou'] = args.giou_loss_coef
    weight_dict.update(args.other_loss_coefs)
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    weight_dict['pseudo_loss'] = args.pseudo_strategy_loss_coef
    weight_dict['div_loss'] = args.diversity_loss_coef

    criterion = VGCriterion(weight_dict=weight_dict, loss_loc=args.loss_loc, box_xyxy=args.box_xyxy)
    criterion.to(device)

    postprocessor = PostProcess(args.box_xyxy)

    return model, criterion, postprocessor


class GroundTextBlock_multihop(nn.Module):
    def __init__(self, NFilm=2, with_residual=True, textdim=768,visudim=512,emb_size=512,fusion='cat',intmd=False,lstm=False,erasing=0.):
        super(GroundTextBlock_multihop, self).__init__()

        self.NFilm = NFilm
        self.emb_size = emb_size
        self.with_residual = with_residual
        self.cont_size = emb_size
        self.fusion = fusion
        self.intmd = intmd
        self.lstm = lstm
        self.erasing = erasing
        if self.fusion == 'cat':
            self.cont_size = emb_size * 2

        self.modulesdict = nn.ModuleDict()
        modules = OrderedDict()
        modules["film0"] = GroundTextBlock_context(textdim=textdim, visudim=emb_size, contextdim=emb_size,
                                                   emb_size=emb_size, fusion=fusion, lstm=self.lstm)
        for n in range(1, NFilm):
            modules["conv%d" % n] = ConvBatchNormReLU(emb_size, emb_size, 3, 1, 1, 1)
            modules["film%d" % n] = GroundTextBlock_context(textdim=textdim, visudim=emb_size,
                                                            contextdim=self.cont_size, emb_size=emb_size, fusion=fusion,
                                                            lstm=self.lstm)
        self.modulesdict.update(modules)

        self.MLP1 = torch.nn.Sequential(
            nn.Linear(emb_size * 3, emb_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(emb_size, 2),
            nn.ReLU(), )

    def forward(self, fvisu, fword, weight=None, fsent=None, word_mask=None):
        B, Dvisu, H, W = fvisu.size()
        B, N, _ = fword.size()
        intmd_feat, attnscore_list = [], []

        x, attn_lang, attn_score = self.modulesdict["film0"](fvisu, fword, torch.ones(B, N).cuda(), fsent=fsent, word_mask=word_mask)
        fmodu_multi = x
        attn_lang_multi = attn_lang
        attnscore_list.append(attn_score.view(B, N, 1, 1))
        if self.intmd:
            intmd_feat.append(x)
        if self.NFilm==1:
            intmd_feat = [x]
        for n in range(1,self.NFilm):
            score_list = [mask_softmax(score.squeeze(2).squeeze(2),word_mask,lstm=self.lstm) for score in attnscore_list]
            score = torch.clamp(torch.max(torch.stack(score_list, dim=1), dim=1, keepdim=False)[0],min=0.,max=1.)
            x = self.modulesdict["conv%d"%n](x)
            x, attn_lang, attn_score = self.modulesdict["film%d"%n](x, fword, (1-score), fsent=fsent,word_mask=word_mask)
            fmodu_multi = torch.cat((fmodu_multi, x), dim=1)
            attn_lang_multi = torch.cat((attn_lang_multi, attn_lang), dim=1)
            attnscore_list.append(attn_score.view(B,N,1,1)) ## format match div loss in main func
            if self.intmd:
                intmd_feat.append(x)
            elif n == self.NFilm-1:  # this branch
                intmd_feat = [x]

        # Segment Activation
        visu_multi = fmodu_multi.view(B, -1, H*W).contiguous().permute(0,2,1)
        lang_multi = attn_lang_multi.permute(0,2,1)
        visu_lang_cross_attn_feat, visu_lang_cross_attn_score = func_attention(lang_multi, visu_multi, raw_feature_norm='softmax')
        # visu_lang_cross_attn_feat_final = self.MLP1(visu_lang_cross_attn_feat).squeeze()  # TODO: add number
        visu_lang_cross_attn_feat_final = self.MLP1(visu_lang_cross_attn_feat)
        visu_lang_cross_attn_feat_final = F.softmax(visu_lang_cross_attn_feat_final, dim=2)

        word_mask_wo_cls_sep = word_mask.clone()
        word_mask_wo_cls_sep[:, 0] = 0
        for i in range(B):
            word_mask_wo_cls_sep[i][len(torch.nonzero(word_mask[i]))-1] = 0
        assert len(torch.nonzero(word_mask_wo_cls_sep)) == len(torch.nonzero(word_mask)) - 2 * B

        lang_visu_score = visu_lang_cross_attn_feat_final[:, :, 1] * word_mask_wo_cls_sep
        lang_fact_score = visu_lang_cross_attn_feat_final[:, :, 0] * word_mask_wo_cls_sep

        return lang_visu_score, lang_fact_score, attnscore_list, fmodu_multi


class GroundTextBlock_context(nn.Module):
    def __init__(self, with_residual=True, textdim=768,visudim=512,contextdim=512,emb_size=512,fusion='prod',cont_map=False, lstm=False,baseline=False):
        super(GroundTextBlock_context, self).__init__()

        self.cont_map = cont_map    ## mapping context with language feature
        self.lstm = lstm
        self.emb_size = emb_size
        self.with_residual = with_residual
        self.fusion = fusion
        self.baseline = baseline
        self.film = FiLM()

        if self.cont_map:
            self.sent_map = nn.Linear(768, emb_size)
            self.context_map = nn.Linear(emb_size, emb_size)
        if self.fusion == 'cat':
            self.attn_map = nn.Conv1d(textdim+visudim, emb_size//2, kernel_size=1)
        elif self.fusion == 'prod':
            assert(textdim==visudim) ## if product fusion
            self.attn_map = nn.Conv1d(visudim, emb_size//2, kernel_size=1)

        self.attn_score = nn.Conv1d(emb_size//2, 1, kernel_size=1)
        if self.baseline:
            self.fusion_layer = ConvBatchNormReLU(visudim+textdim+8, emb_size, 1, 1, 0, 1)
        else:
            self.gamme_decode = nn.Linear(textdim, 2 * emb_size)
            self.conv1 = nn.Conv2d(visudim, emb_size, kernel_size=1)
            # self.bn1 = nn.BatchNorm2d(emb_size)
            self.bn1 = nn.InstanceNorm2d(emb_size)
        init_modules(self.modules())


    def forward(self, fvisu, fword, context_score, textattn=None,weight=None,fsent=None,word_mask=None):
        fword = fword.permute(0, 2, 1)
        B, Dvisu, H, W = fvisu.size()
        B, Dlang, N = fword.size()
        B, N = context_score.size()
        assert(Dvisu==Dlang)

        if self.cont_map and fsent is not None:
            fsent = F.normalize(F.relu(self.sent_map(fsent)), p=2, dim=1)
            fcont = torch.matmul(context_score.view(B,1,N),fword.permute(0,2,1)).squeeze(1)
            fcontext = F.relu(self.context_map(fsent*fcont)).unsqueeze(2).repeat(1,1,N)
            ## word attention
            tile_visu = torch.mean(fvisu.view(B, Dvisu, -1),dim=2,keepdim=True).repeat(1,1,N)
            if self.fusion == 'cat':
                context_tile = torch.cat([tile_visu,\
                    fword, fcontext], dim=1)
            elif self.fusion == 'prod':
                context_tile = tile_visu * \
                    fword * fcontext
        else:  # this branch
            ## word attention
            tile_visu = torch.mean(fvisu.view(B, Dvisu, -1),dim=2,keepdim=True).repeat(1,1,N)
            if self.fusion == 'cat':
                context_tile = torch.cat([tile_visu,\
                    fword * context_score.view(B, 1, N).repeat(1, Dlang, 1,)], dim=1)
            elif self.fusion == 'prod':  # this branch
                context_tile = tile_visu * \
                    fword * context_score.view(B, 1, N).repeat(1, Dlang, 1,)

        attn_feat = torch.tanh(self.attn_map(context_tile))
        attn_score = self.attn_score(attn_feat).squeeze(1)

        mask_score = mask_softmax(attn_score,word_mask,lstm=self.lstm)
        attn_lang_my = mask_score.view(B,1,N) * fword   # [8,512,20]
        attn_lang = torch.matmul(mask_score.view(B,1,N),fword.permute(0,2,1))  # [8,512]
        attn_lang = attn_lang.view(B,Dlang).squeeze(1)

        if self.baseline:
            # fmodu = self.fusion_layer(torch.cat([fvisu, attn_lang.unsqueeze(2).unsqueeze(2).repeat(1,1,fvisu.shape[-1],fvisu.shape[-1]),fcoord],dim=1))
            pass
        else:  # this branch
            ## lang-> gamma, beta
            film_param = self.gamme_decode(attn_lang)
            film_param = film_param.view(B,2*self.emb_size,1,1).repeat(1,1,H,W)
            gammas, betas = torch.split(film_param, self.emb_size, dim=1)
            gammas, betas = torch.tanh(gammas), torch.tanh(betas)

            ## modulate visu feature
            # fmodu = self.bn1(self.conv1(torch.cat([fvisu,fcoord],dim=1)))
            fmodu = self.bn1(self.conv1(fvisu))
            fmodu = self.film(fmodu, gammas, betas)
            fmodu = F.relu(fmodu)
        return fmodu, attn_lang_my, attn_score


class FiLM(nn.Module):
    """
    A Feature-wise Linear Modulation Layer from
    'FiLM: Visual Reasoning with a General Conditioning Layer'
    """
    def forward(self, x, gammas, betas):
        # gammas = gammas.unsqueeze(2).unsqueeze(3).expand_as(x)
        # betas = betas.unsqueeze(2).unsqueeze(3).expand_as(x)
        return (gammas * x) + betas


def mask_softmax(attn_score, word_mask, temperature=10., clssep=False, lstm=False):
    if len(attn_score.shape)!=2:
        attn_score = attn_score.squeeze(2).squeeze(2)
    word_mask_cp = word_mask[:,:attn_score.shape[1]].clone()
    score = F.softmax(attn_score*temperature, dim=1)
    if not clssep:
        for ii in range(word_mask_cp.shape[0]):
            if lstm:
                word_mask_cp[ii,word_mask_cp[ii,:].sum()-1]=0
            else:  # this branch
                word_mask_cp[ii,0]=0
                word_mask_cp[ii,word_mask_cp[ii,:].sum()]=0 ## set one to 0 already
    mask_score = score * word_mask_cp.float()
    mask_score = mask_score/(mask_score.sum(1)+1e-8).view(mask_score.size(0), 1).expand(mask_score.size(0), mask_score.size(1))
    return mask_score


def init_modules(modules, init='uniform'):
    if init.lower() == 'normal':
        init_params = kaiming_normal_
    elif init.lower() == 'uniform':
        init_params = kaiming_uniform_
    else:
        return
    for m in modules:
        if isinstance(m, (nn.Conv3d, nn.Conv2d, nn.Linear)):
            init_params(m.weight)


def func_attention(query, context, raw_feature_norm, smooth=9, eps=1e-8, weight=None):
    """
    query: (n_context, queryL, d)
    context: (n_context, sourceL, d)
    """
    batch_size_q, queryL = query.size(0), query.size(1)
    batch_size, sourceL = context.size(0), context.size(1)

    # Get attention
    # --> (batch, d, queryL)
    queryT = torch.transpose(query, 1, 2)

    # (batch, sourceL, d)(batch, d, queryL)
    # --> (batch, sourceL, queryL)
    attn = torch.bmm(context, queryT)

    if raw_feature_norm == "softmax":
        # --> (batch*sourceL, queryL)
        attn = attn.view(batch_size * sourceL, queryL)
        attn = F.softmax(attn, dim=1)
        # --> (batch, sourceL, queryL)
        attn = attn.view(batch_size, sourceL, queryL)
    elif raw_feature_norm == "l2norm":
        attn = l2norm(attn, 2)
    elif raw_feature_norm == "clipped_l2norm":
        attn = nn.LeakyReLU(0.1)(attn)
        attn = l2norm(attn, 2)
    elif raw_feature_norm == "l1norm":
        attn = l1norm(attn, 2)
    elif raw_feature_norm == "clipped_l1norm":
        attn = nn.LeakyReLU(0.1)(attn)
        attn = l1norm(attn, 2)
    elif raw_feature_norm == "clipped":
        attn = nn.LeakyReLU(0.1)(attn)
    elif raw_feature_norm == "no_norm":
        pass
    else:
        raise ValueError("unknown first norm type:", raw_feature_norm)

    if weight is not None:
        attn = attn + weight

    attn_out = attn.clone()

    # --> (batch, queryL, sourceL)
    attn = torch.transpose(attn, 1, 2).contiguous()
    # --> (batch*queryL, sourceL)
    attn = attn.view(batch_size * queryL, sourceL)

    attn = F.softmax(attn * smooth, dim=1)
    # --> (batch, queryL, sourceL)
    attn = attn.view(batch_size, queryL, sourceL)
    # --> (batch, sourceL, queryL)
    attnT = torch.transpose(attn, 1, 2).contiguous()

    # --> (batch, d, sourceL)
    contextT = torch.transpose(context, 1, 2)
    # (batch x d x sourceL)(batch x sourceL x queryL)
    # --> (batch, d, queryL)
    weightedContext = torch.bmm(contextT, attnT)
    # --> (batch, queryL, d)
    weightedContext = torch.transpose(weightedContext, 1, 2)

    return weightedContext, attn_out


def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X


def l1norm(X, dim, eps=1e-8):
    """L1-normalize columns of X
    """
    norm = torch.abs(X).sum(dim=dim, keepdim=True) + eps
    X = torch.div(X, norm)
    return X


class ConvBatchNormReLU(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        leaky=False,
        relu=True,
        instance=False,
    ):
        super(ConvBatchNormReLU, self).__init__()
        self.add_module(
            "conv",
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
        )
        if instance:
            self.add_module(
                "bn",
                nn.InstanceNorm2d(num_features=out_channels),
            )
        else:
            self.add_module(
                "bn",
                nn.BatchNorm2d(
                    num_features=out_channels, eps=1e-5, momentum=0.999, affine=True
                ),
            )

        if leaky:
            self.add_module("relu", nn.LeakyReLU(0.1))
        elif relu:
            self.add_module("relu", nn.ReLU())

    def forward(self, x):
        return super(ConvBatchNormReLU, self).forward(x)