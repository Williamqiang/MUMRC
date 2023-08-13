import numpy as np
import mindspore
import mindspore.numpy as ms_np
import mindspore.ops as P
from mindspore import nn
from mindspore import Tensor, Parameter


class GELU(nn.Cell):

    def __init__(self):
        super().__init__()
        self.erf = P.Erf()
        self.sqrt = P.Sqrt()
        self.const0 = Tensor(0.5, mindspore.float32)
        self.const1 = Tensor(1.0, mindspore.float32)
        self.const2 = Tensor(2.0, mindspore.float32)

    def construct(self, x):
        return x * self.const0 * (self.const1 + self.erf(x / self.sqrt(self.const2)))


class BertEmbeddings(nn.Cell):

    def __init__(self):
        super(BertEmbeddings, self).__init__()
        self.pattern2_0_pattern_weight_0 = Tensor(np.random.uniform(0, 1, (1, 512)).astype(np.int64))
                                                  
        self.embedding_1 = nn.Embedding(vocab_size=30522, embedding_size=768, padding_idx=0)
        self.embedding_2 = nn.Embedding(vocab_size=2, embedding_size=768, padding_idx=None)
        self.embedding_3 = nn.Embedding(vocab_size=512, embedding_size=768, padding_idx=None)
        self.add_4_alpha = 1
        self.add__5_alpha = 1
        self.layernorm_6 = nn.LayerNorm(normalized_shape=(768, ), epsilon=1e-12)
        self.dropout_7 = nn.Dropout(p=0.1)

    def construct(self, x, x0):
        _, opt_pattern2 = x.shape
        opt_pattern2 = opt_pattern2 + 0
        opt_pattern2_slice_0 = self.pattern2_0_pattern_weight_0[0:1:1]
        opt_pattern2_0 = opt_pattern2_slice_0[:, 0:opt_pattern2:1]
        opt_embedding_1 = self.embedding_1(x)
        opt_embedding_2 = self.embedding_2(x0)
        opt_embedding_3 = self.embedding_3(opt_pattern2_0)
        opt_add_4 = opt_embedding_1 + opt_embedding_2
        opt_add__5 = opt_add_4 + opt_embedding_3 * self.add__5_alpha
        opt_layernorm_6 = self.layernorm_6(opt_add__5)
        opt_dropout_7 = self.dropout_7(opt_layernorm_6)
        return opt_dropout_7


class BertSelfAttention(nn.Cell):

    def __init__(self):
        super(BertSelfAttention, self).__init__()
        self.transpose_0_input = Parameter(Tensor(np.random.uniform(0, 1, (768, 768)).astype(np.float32)), name=None)
        self.transpose_1_input = Parameter(Tensor(np.random.uniform(0, 1, (768, 768)).astype(np.float32)), name=None)
        self.transpose_2_input = Parameter(Tensor(np.random.uniform(0, 1, (768, 768)).astype(np.float32)), name=None)
        self.add__6_alpha = 1
        self.add__6_input_1 = Parameter(Tensor(np.random.uniform(0, 1, (768, )).astype(np.float32)), name=None)
        self.add__7_alpha = 1
        self.add__7_input_1 = Parameter(Tensor(np.random.uniform(0, 1, (768, )).astype(np.float32)), name=None)
        self.add__8_alpha = 1
        self.add__8_input_1 = Parameter(Tensor(np.random.uniform(0, 1, (768, )).astype(np.float32)), name=None)
        self.transpose_12_dims = (0, 2, 1, 3)
        self.transpose_13_dims = (0, 2, 1, 3)
        self.transpose_14_dims = (0, 2, 1, 3)
        self.transpose_15_input_perm = (0, 1, 3, 2)
        self.div_17_input_1 = 8.0
        self.add_18_alpha = 1
        self.softmax_19 = nn.Softmax(axis=-1)
        self.dropout_20 = nn.Dropout(p=0.1)
        self.transpose_22_dims = (0, 2, 1, 3)

    def construct(self, x, x0):
        opt_transpose_0 = self.transpose_0_input.transpose(1, 0)
        opt_transpose_1 = self.transpose_1_input.transpose(1, 0)
        opt_transpose_2 = self.transpose_2_input.transpose(1, 0)
        opt_matmul_3 = P.matmul(x, opt_transpose_0)
        opt_matmul_4 = P.matmul(x, opt_transpose_1)
        opt_matmul_5 = P.matmul(x, opt_transpose_2)
        opt_add__6 = opt_matmul_3 + self.add__6_input_1 * self.add__6_alpha
        opt_add__7 = opt_matmul_4 + self.add__7_input_1 * self.add__7_alpha
        opt_add__8 = opt_matmul_5 + self.add__8_input_1 * self.add__8_alpha
        x_0, x_1, _ = opt_add__6.shape
        opt_add__6_ = opt_add__6.view(x_0, x_1, 12, 64)
        opt_pattern1_9 = opt_add__6_
        x_0, x_1, _ = opt_add__7.shape
        opt_add__7_ = opt_add__7.view(x_0, x_1, 12, 64)
        opt_pattern1_10 = opt_add__7_
        x_0, x_1, _ = opt_add__8.shape
        opt_add__8_ = opt_add__8.view(x_0, x_1, 12, 64)
        opt_pattern1_11 = opt_add__8_
        opt_transpose_12 = opt_pattern1_9.transpose(*self.transpose_12_dims)
        opt_transpose_13 = opt_pattern1_10.transpose(*self.transpose_13_dims)
        opt_transpose_14 = opt_pattern1_11.transpose(*self.transpose_14_dims)
        opt_transpose_15 = P.Transpose()(opt_transpose_13, self.transpose_15_input_perm)
        opt_matmul_16 = P.matmul(opt_transpose_12, opt_transpose_15)
        opt_div_17 = opt_matmul_16 / self.div_17_input_1
        opt_add_18 = opt_div_17 + x0
        opt_softmax_19 = self.softmax_19(opt_add_18)
        opt_dropout_20 = self.dropout_20(opt_softmax_19)
        opt_matmul_21 = P.matmul(opt_dropout_20, opt_transpose_14)
        opt_transpose_22 = opt_matmul_21.transpose(*self.transpose_22_dims)
        x_0, x_1, _, _ = opt_transpose_22.shape
        opt_transpose_22_ = opt_transpose_22.view(x_0, x_1, 768)
        opt_pattern1_23 = opt_transpose_22_
        return opt_pattern1_23


class BertSelfOutput(nn.Cell):

    def __init__(self):
        super(BertSelfOutput, self).__init__()
        self.transpose_0_input = Parameter(Tensor(np.random.uniform(0, 1, (768, 768)).astype(np.float32)), name=None)
        self.add__2_alpha = 1
        self.add__2_input_1 = Parameter(Tensor(np.random.uniform(0, 1, (768, )).astype(np.float32)), name=None)
        self.dropout_3 = nn.Dropout(p=0.1)
        self.add_4_alpha = 1
        self.layernorm_5 = nn.LayerNorm(normalized_shape=(768, ), epsilon=1e-12)

    def construct(self, x, x0):
        opt_transpose_0 = self.transpose_0_input.transpose(1, 0)
        opt_matmul_1 = P.matmul(x, opt_transpose_0)
        opt_add__2 = opt_matmul_1 + self.add__2_input_1 * self.add__2_alpha
        opt_dropout_3 = self.dropout_3(opt_add__2)
        opt_add_4 = opt_dropout_3 + x0
        opt_layernorm_5 = self.layernorm_5(opt_add_4)
        return opt_layernorm_5


class BertAttention(nn.Cell):

    def __init__(self):
        super(BertAttention, self).__init__()
        self.bertselfattention_0 = BertSelfAttention()
        self.bertselfoutput_0 = BertSelfOutput()

    def construct(self, x, x0):
        bertselfattention_0_opt = self.bertselfattention_0(x, x0)
        bertselfoutput_0_opt = self.bertselfoutput_0(bertselfattention_0_opt, x)
        return bertselfoutput_0_opt


class BertIntermediate(nn.Cell):

    def __init__(self):
        super(BertIntermediate, self).__init__()
        self.transpose_0_input = Parameter(Tensor(np.random.uniform(0, 1, (3072, 768)).astype(np.float32)), name=None)
        self.add__2_alpha = 1
        self.add__2_input_1 = Parameter(Tensor(np.random.uniform(0, 1, (3072, )).astype(np.float32)), name=None)
        self.gelu_3 = GELU()

    def construct(self, x):
        opt_transpose_0 = self.transpose_0_input.transpose(1, 0)
        opt_matmul_1 = P.matmul(x, opt_transpose_0)
        opt_add__2 = opt_matmul_1 + self.add__2_input_1 * self.add__2_alpha
        opt_gelu_3 = self.gelu_3(opt_add__2)
        return opt_gelu_3


class BertOutput(nn.Cell):

    def __init__(self):
        super(BertOutput, self).__init__()
        self.transpose_0_input = Parameter(Tensor(np.random.uniform(0, 1, (768, 3072)).astype(np.float32)), name=None)
        self.add__2_alpha = 1
        self.add__2_input_1 = Parameter(Tensor(np.random.uniform(0, 1, (768, )).astype(np.float32)), name=None)
        self.dropout_3 = nn.Dropout(p=0.1)
        self.add_4_alpha = 1
        self.layernorm_5 = nn.LayerNorm(normalized_shape=(768, ), epsilon=1e-12)

    def construct(self, x, x0):
        opt_transpose_0 = self.transpose_0_input.transpose(1, 0)
        opt_matmul_1 = P.matmul(x, opt_transpose_0)
        opt_add__2 = opt_matmul_1 + self.add__2_input_1 * self.add__2_alpha
        opt_dropout_3 = self.dropout_3(opt_add__2)
        opt_add_4 = opt_dropout_3 + x0
        opt_layernorm_5 = self.layernorm_5(opt_add_4)
        return opt_layernorm_5


class BertLayer(nn.Cell):

    def __init__(self):
        super(BertLayer, self).__init__()
        self.bertattention_0 = BertAttention()
        self.bertintermediate_0 = BertIntermediate()
        self.bertoutput_0 = BertOutput()

    def construct(self, x, x0):
        bertattention_0_opt = self.bertattention_0(x, x0)
        bertintermediate_0_opt = self.bertintermediate_0(bertattention_0_opt)
        bertoutput_0_opt = self.bertoutput_0(bertintermediate_0_opt, bertattention_0_opt)
        return bertoutput_0_opt


class Layer(nn.Cell):

    def __init__(self):
        super(Layer, self).__init__()
        self.bertlayer_11 = BertLayer()
        self.bertlayer_4 = BertLayer()
        self.bertlayer_6 = BertLayer()
        self.bertlayer_10 = BertLayer()
        self.bertlayer_3 = BertLayer()
        self.bertlayer_8 = BertLayer()
        self.bertlayer_0 = BertLayer()
        self.bertlayer_9 = BertLayer()
        self.bertlayer_2 = BertLayer()
        self.bertlayer_1 = BertLayer()
        self.bertlayer_5 = BertLayer()
        self.bertlayer_7 = BertLayer()

    def construct(self, x, x0):
        bertlayer_11_opt = self.bertlayer_11(x, x0)
        bertlayer_4_opt = self.bertlayer_4(bertlayer_11_opt, x0)
        bertlayer_6_opt = self.bertlayer_6(bertlayer_4_opt, x0)
        bertlayer_10_opt = self.bertlayer_10(bertlayer_6_opt, x0)
        bertlayer_3_opt = self.bertlayer_3(bertlayer_10_opt, x0)
        bertlayer_8_opt = self.bertlayer_8(bertlayer_3_opt, x0)
        bertlayer_0_opt = self.bertlayer_0(bertlayer_8_opt, x0)
        bertlayer_9_opt = self.bertlayer_9(bertlayer_0_opt, x0)
        bertlayer_2_opt = self.bertlayer_2(bertlayer_9_opt, x0)
        bertlayer_1_opt = self.bertlayer_1(bertlayer_2_opt, x0)
        bertlayer_5_opt = self.bertlayer_5(bertlayer_1_opt, x0)
        bertlayer_7_opt = self.bertlayer_7(bertlayer_5_opt, x0)
        return bertlayer_7_opt


class BertEncoder(nn.Cell):

    def __init__(self):
        super(BertEncoder, self).__init__()
        self.layer_0 = Layer()

    def construct(self, x, x0):
        layer_0_opt = self.layer_0(x, x0)
        return layer_0_opt


class BertModel(nn.Cell):

    def __init__(self):
        super(BertModel, self).__init__()
        self.prim_slice_0_starts = 0
        self.prim_slice_0_steps = 1
        self.expanddims_1 = P.ExpandDims()
        self.expanddims_1_axis = 1
        self.prim_slice_2_starts = 0
        self.prim_slice_2_steps = 1
        self.prim_slice_3_starts = 0
        self.prim_slice_3_steps = 1
        self.cast_4_var = mindspore.float32
        self.rsub_5_input_1 = 1.0
        self.rsub_5_alpha = 1
        self.mul_6_input_1 = -10000.0
        self.bertembeddings_0 = BertEmbeddings()
        self.bertencoder_0 = BertEncoder()

    def construct(self, x, x0, x1):
        opt_prim_slice_0 = x[self.prim_slice_0_starts::self.prim_slice_0_steps, :, :]
        opt_expanddims_1 = self.expanddims_1(opt_prim_slice_0, self.expanddims_1_axis)
        opt_prim_slice_2 = opt_expanddims_1[:, :, self.prim_slice_2_starts::self.prim_slice_2_steps, :]
        opt_prim_slice_3 = opt_prim_slice_2[:, :, :, self.prim_slice_3_starts::self.prim_slice_3_steps]
        opt_cast_4 = P.Cast()(opt_prim_slice_3, self.cast_4_var)
        opt_rsub_5 = self.rsub_5_input_1 - opt_cast_4 * self.rsub_5_alpha
        opt_mul_6 = opt_rsub_5 * self.mul_6_input_1
        bertembeddings_0_opt = self.bertembeddings_0(x0, x1)
        bertencoder_0_opt = self.bertencoder_0(bertembeddings_0_opt, opt_mul_6)
        return bertencoder_0_opt


class CLIPVisionEmbeddings(nn.Cell):

    def __init__(self):
        super(CLIPVisionEmbeddings, self).__init__()
        # self.randn_0 = aten.randn()

        self.class_embedding = Parameter(Tensor(np.random.uniform(0, 1, (1, 768)).astype(np.float32)), name=None)

        self.patch_embedding = nn.Conv2d(
            in_channels=3, out_channels=768, kernel_size=32, stride=32
        )
        # self.position_ids=mindspore.Tensor(np.arange(48).reshape(1, -1).astype(np.int32))

        self.aux_position_embedding = nn.Embedding(3*4*4, 768)
        self.aux_position_ids=mindspore.Tensor(np.arange(48).reshape(1, -1).astype(np.int32),const_arg =True)
        self.aux_position_ids=self.aux_position_ids.reshape((1, -1))
        # print(self.aux_position_ids)

        self.rcnn_position_embedding = nn.Embedding(3*2*2, 768)
        self.rcnn_position_ids=mindspore.Tensor(np.arange(12).reshape(1, -1).astype(np.int32),const_arg =True)
        self.rcnn_position_ids=self.rcnn_position_ids.reshape((1, -1))

    def construct(self,pixel_values, aux_embeddings, rcnn_embeddings):
        batch_size = aux_embeddings.shape[0]
        # print(batch_size)
        # exit()
        class_embeds = self.class_embedding[:,None,:].repeat(batch_size,axis=0 )
        # print(class_embeds.shape)
        # exit()
        embeddings = class_embeds
        aux_embeds = []
        for aux_embedding in aux_embeddings:
            # print(aux_embedding)
            aux_embedding=aux_embedding.astype(np.float32)
            aux_embed = self.patch_embedding(aux_embedding)
            # exit()
            # print(P.shape(aux_embed))
            aux_embed = aux_embed.flatten(start_dim=2).transpose((0,2, 1)).flatten(start_dim=0, end_dim=1)    # 3*16, 768 3个子图
            aux_embeds.append(aux_embed)
        aux_embeds = P.stack(aux_embeds) # bsz, 48, 768
        aux_embeds = aux_embeds + self.aux_position_embedding(self.aux_position_ids)
        embeddings = P.cat((embeddings, aux_embeds), axis=1)

        rcnn_embeds = []
        for rcnn_embedding in rcnn_embeddings:
            rcnn_embedding=rcnn_embedding.astype(np.float32)
            rcnn_embed = self.patch_embedding(rcnn_embedding)
            rcnn_embed = rcnn_embed.flatten(start_dim=2).transpose((0,2, 1)).flatten(start_dim=0, end_dim=1)   # 3*4, 768
            rcnn_embeds.append(rcnn_embed)
        rcnn_embeds = P.stack(rcnn_embeds) # bsz, 12, 768
        rcnn_embeds = rcnn_embeds + self.rcnn_position_embedding(self.rcnn_position_ids)
        embeddings = P.cat((embeddings, rcnn_embeds), axis=1)
        return embeddings


class CLIPAttention(nn.Cell):

    def __init__(self):
        super(CLIPAttention, self).__init__()
        self.transpose_0_input = Parameter(Tensor(np.random.uniform(0, 1, (768, 768)).astype(np.float32)), name=None)
        self.transpose_1_input = Parameter(Tensor(np.random.uniform(0, 1, (768, 768)).astype(np.float32)), name=None)
        self.transpose_2_input = Parameter(Tensor(np.random.uniform(0, 1, (768, 768)).astype(np.float32)), name=None)
        self.transpose_3_input = Parameter(Tensor(np.random.uniform(0, 1, (768, 768)).astype(np.float32)), name=None)
        self.mul_10_input_1 = 12
        self.add__11_alpha = 1
        self.add__11_input_1 = Parameter(Tensor(np.random.uniform(0, 1, (768, )).astype(np.float32)), name=None)
        self.add__12_alpha = 1
        self.add__12_input_1 = Parameter(Tensor(np.random.uniform(0, 1, (768, )).astype(np.float32)), name=None)
        self.add__13_alpha = 1
        self.add__13_input_1 = Parameter(Tensor(np.random.uniform(0, 1, (768, )).astype(np.float32)), name=None)
        self.mul_14_input_1 = 0.125
        self.view_15_shape1 = -1
        self.view_15_shape2 = 12
        self.view_15_shape3 = 64
        self.view_16_shape1 = -1
        self.view_16_shape2 = 12
        self.view_16_shape3 = 64
        self.view_17_shape2 = 12
        self.view_17_shape3 = 64
        self.transpose_18_input_perm = (0, 2, 1, 3)
        self.transpose_19_input_perm = (0, 2, 1, 3)
        self.transpose_20_input_perm = (0, 2, 1, 3)
        self.view_21_shape1 = -1
        self.view_21_shape2 = 64
        self.view_22_shape1 = -1
        self.view_22_shape2 = 64
        self.view_23_shape1 = -1
        self.view_23_shape2 = 64
        self.transpose_24_input_perm = (0, 2, 1)
        # self.bmm_25 = aten.bmm()
        self.softmax_26 = nn.Softmax(axis=-1)
        self.dropout_27 = nn.Dropout(p=0.1)
        # self.bmm_28 = aten.bmm()
        self.view_29_shape1 = 12
        self.view_29_shape3 = 64
        self.transpose_30_input_perm = (0, 2, 1, 3)
        self.add__33_alpha = 1
        self.add__33_input_1 = Parameter(Tensor(np.random.uniform(0, 1, (768, )).astype(np.float32)), name=None)

    def construct(self, x):
        opt_transpose_0 = self.transpose_0_input.transpose(1, 0)
        opt_transpose_1 = self.transpose_1_input.transpose(1, 0)
        opt_transpose_2 = self.transpose_2_input.transpose(1, 0)
        opt_transpose_3 = self.transpose_3_input.transpose(1, 0)
        opt_shape_4 = P.Shape()(x)[0]
        opt_shape_5 = P.Shape()(x)[1]
        opt_shape_6 = P.Shape()(x)[2]
        opt_matmul_7 = P.matmul(x, opt_transpose_0)
        opt_matmul_8 = P.matmul(x, opt_transpose_1)
        opt_matmul_9 = P.matmul(x, opt_transpose_2)
        opt_mul_10 = opt_shape_4 * self.mul_10_input_1
        opt_add__11 = opt_matmul_7 + self.add__11_input_1 * self.add__11_alpha
        opt_add__12 = opt_matmul_8 + self.add__12_input_1 * self.add__12_alpha
        opt_add__13 = opt_matmul_9 + self.add__13_input_1 * self.add__13_alpha
        opt_mul_14 = opt_add__11 * self.mul_14_input_1
        opt_view_15 = opt_add__12.view(opt_shape_4, self.view_15_shape1, self.view_15_shape2, self.view_15_shape3)
        opt_view_16 = opt_add__13.view(opt_shape_4, self.view_16_shape1, self.view_16_shape2, self.view_16_shape3)
        opt_view_17 = opt_mul_14.view(opt_shape_4, opt_shape_5, self.view_17_shape2, self.view_17_shape3)
        opt_transpose_18 = P.Transpose()(opt_view_15, self.transpose_18_input_perm)
        opt_transpose_19 = P.Transpose()(opt_view_16, self.transpose_19_input_perm)
        opt_transpose_20 = P.Transpose()(opt_view_17, self.transpose_20_input_perm)
        opt_view_21 = opt_transpose_18.view(opt_mul_10, self.view_21_shape1, self.view_21_shape2)
        opt_view_22 = opt_transpose_19.view(opt_mul_10, self.view_22_shape1, self.view_22_shape2)
        opt_view_23 = opt_transpose_20.view(opt_mul_10, self.view_23_shape1, self.view_23_shape2)
        opt_transpose_24 = P.Transpose()(opt_view_21, self.transpose_24_input_perm)
        opt_bmm_25 = P.bmm(opt_view_23, opt_transpose_24)
        opt_softmax_26 = self.softmax_26(opt_bmm_25)
        opt_dropout_27 = self.dropout_27(opt_softmax_26)
        opt_bmm_28 = P.bmm(opt_dropout_27, opt_view_22)
        opt_view_29 = opt_bmm_28.view(opt_shape_4, self.view_29_shape1, opt_shape_5, self.view_29_shape3)
        opt_transpose_30 = P.Transpose()(opt_view_29, self.transpose_30_input_perm)
        opt_reshape_31 = opt_transpose_30.reshape(opt_shape_4, opt_shape_5, opt_shape_6)
        opt_matmul_32 = P.matmul(opt_reshape_31, opt_transpose_3)
        opt_add__33 = opt_matmul_32 + self.add__33_input_1 * self.add__33_alpha
        return opt_add__33


class QuickGELUActivation(nn.Cell):

    def __init__(self):
        super(QuickGELUActivation, self).__init__()
        self.mul_0_input_1 = 1.702
        self.sigmoid_1 = nn.Sigmoid()

    def construct(self, x):
        opt_mul_0 = x * self.mul_0_input_1
        opt_sigmoid_1 = self.sigmoid_1(opt_mul_0)
        opt_mul_2 = x * opt_sigmoid_1
        return opt_mul_2


class CLIPMLP(nn.Cell):

    def __init__(self):
        super(CLIPMLP, self).__init__()
        self.transpose_0_input = Parameter(Tensor(np.random.uniform(0, 1, (3072, 768)).astype(np.float32)), name=None)
        self.add__3_alpha = 1
        self.add__3_input_1 = Parameter(Tensor(np.random.uniform(0, 1, (3072, )).astype(np.float32)), name=None)
        self.quickgeluactivation_0 = QuickGELUActivation()
        self.transpose_1_input = Parameter(Tensor(np.random.uniform(0, 1, (768, 3072)).astype(np.float32)), name=None)
        self.add__5_alpha = 1
        self.add__5_input_1 = Parameter(Tensor(np.random.uniform(0, 1, (768, )).astype(np.float32)), name=None)

    def construct(self, x):
        opt_transpose_0 = self.transpose_0_input.transpose(1, 0)
        opt_matmul_2 = P.matmul(x, opt_transpose_0)
        opt_add__3 = opt_matmul_2 + self.add__3_input_1 * self.add__3_alpha
        quickgeluactivation_0_opt = self.quickgeluactivation_0(opt_add__3)
        opt_transpose_1 = self.transpose_1_input.transpose(1, 0)
        opt_matmul_4 = P.matmul(quickgeluactivation_0_opt, opt_transpose_1)
        opt_add__5 = opt_matmul_4 + self.add__5_input_1 * self.add__5_alpha
        return opt_add__5


class CLIPEncoderLayer(nn.Cell):

    def __init__(self):
        super(CLIPEncoderLayer, self).__init__()
        self.layernorm_0 = nn.LayerNorm(normalized_shape=(768, ), epsilon=1e-05)
        self.clipattention_0 = CLIPAttention()
        self.add_1_alpha = 1
        self.layernorm_2 = nn.LayerNorm(normalized_shape=(768, ), epsilon=1e-05)
        self.clipmlp_0 = CLIPMLP()
        self.add_3_alpha = 1

    def construct(self, x):
        opt_layernorm_0 = self.layernorm_0(x)
        clipattention_0_opt = self.clipattention_0(opt_layernorm_0)
        opt_add_1 = x + clipattention_0_opt
        opt_layernorm_2 = self.layernorm_2(opt_add_1)
        clipmlp_0_opt = self.clipmlp_0(opt_layernorm_2)
        opt_add_3 = opt_add_1 + clipmlp_0_opt
        return opt_add_3


class Layers(nn.Cell):

    def __init__(self):
        super(Layers, self).__init__()
        self.clipencoderlayer_11 = CLIPEncoderLayer()
        self.clipencoderlayer_4 = CLIPEncoderLayer()
        self.clipencoderlayer_0 = CLIPEncoderLayer()
        self.clipencoderlayer_8 = CLIPEncoderLayer()
        self.clipencoderlayer_9 = CLIPEncoderLayer()
        self.clipencoderlayer_5 = CLIPEncoderLayer()
        self.clipencoderlayer_3 = CLIPEncoderLayer()
        self.clipencoderlayer_6 = CLIPEncoderLayer()
        self.clipencoderlayer_7 = CLIPEncoderLayer()
        self.clipencoderlayer_10 = CLIPEncoderLayer()
        self.clipencoderlayer_1 = CLIPEncoderLayer()
        self.clipencoderlayer_2 = CLIPEncoderLayer()

    def construct(self, x):
        clipencoderlayer_11_opt = self.clipencoderlayer_11(x)
        clipencoderlayer_4_opt = self.clipencoderlayer_4(clipencoderlayer_11_opt)
        clipencoderlayer_0_opt = self.clipencoderlayer_0(clipencoderlayer_4_opt)
        clipencoderlayer_8_opt = self.clipencoderlayer_8(clipencoderlayer_0_opt)
        clipencoderlayer_9_opt = self.clipencoderlayer_9(clipencoderlayer_8_opt)
        clipencoderlayer_5_opt = self.clipencoderlayer_5(clipencoderlayer_9_opt)
        clipencoderlayer_3_opt = self.clipencoderlayer_3(clipencoderlayer_5_opt)
        clipencoderlayer_6_opt = self.clipencoderlayer_6(clipencoderlayer_3_opt)
        clipencoderlayer_7_opt = self.clipencoderlayer_7(clipencoderlayer_6_opt)
        clipencoderlayer_10_opt = self.clipencoderlayer_10(clipencoderlayer_7_opt)
        clipencoderlayer_1_opt = self.clipencoderlayer_1(clipencoderlayer_10_opt)
        clipencoderlayer_2_opt = self.clipencoderlayer_2(clipencoderlayer_1_opt)
        return clipencoderlayer_2_opt


class CLIPEncoder(nn.Cell):

    def __init__(self):
        super(CLIPEncoder, self).__init__()
        self.layers_0 = Layers()

    def construct(self, x):
        layers_0_opt = self.layers_0(x)
        return layers_0_opt


class CLIPVisionTransformer(nn.Cell):

    def __init__(self):
        super(CLIPVisionTransformer, self).__init__()
        self.clipvisionembeddings_0 = CLIPVisionEmbeddings()
        self.layernorm_0 = nn.LayerNorm(normalized_shape=(768, ), epsilon=1e-05)
        self.clipencoder_0 = CLIPEncoder()

    def construct(self,pixel_values,aux_values,rcnn_values):
        clipvisionembeddings_0_opt = self.clipvisionembeddings_0(pixel_values,aux_values,rcnn_values)
        opt_layernorm_0 = self.layernorm_0(clipvisionembeddings_0_opt)
        clipencoder_0_opt = self.clipencoder_0(opt_layernorm_0)
        return clipencoder_0_opt


class VisualBertEncoder(nn.Cell):

    def __init__(self):
        super(VisualBertEncoder, self).__init__()
        self.layer_0 = Layer()

    def construct(self, x, x0):
        layer_0_opt = self.layer_0(x, x0)
        return layer_0_opt


class VisualEncoder(nn.Cell):

    def __init__(self):
        super(VisualEncoder, self).__init__()
        self.bertmodel_0 = BertModel()
        self.clipvisiontransformer_0 = CLIPVisionTransformer()
        self.concat_1139 = P.Concat(axis=1)
        self.prim_slice_149_starts = 0
        self.prim_slice_149_steps = 1
        self.expanddims_226 = P.ExpandDims()
        self.expanddims_226_axis = 1
        self.prim_slice_230_starts = 0
        self.prim_slice_230_steps = 1
        self.prim_slice_239_starts = 0
        self.prim_slice_239_steps = 1
        self.rsub_246_input_1 = 1.0
        self.rsub_246_alpha = 1
        self.mul_254_input_1 = -10000.0
        self.visualbertencoder_0 = VisualBertEncoder()

    def construct(self, input_ids, attention_mask,token_type_ids,attention_with_image,pixel_values,aux_values,rcnn_values):
        bertmodel_0_opt = self.bertmodel_0(attention_mask, input_ids, token_type_ids)
        clipvisiontransformer_0_opt = self.clipvisiontransformer_0(pixel_values,aux_values,rcnn_values)
        opt_concat_1139 = self.concat_1139((bertmodel_0_opt, clipvisiontransformer_0_opt))
        opt_prim_slice_149 = attention_with_image[self.prim_slice_149_starts::self.prim_slice_149_steps, :, :]
        opt_expanddims_226 = self.expanddims_226(opt_prim_slice_149, self.expanddims_226_axis)
        opt_prim_slice_230 = opt_expanddims_226[:, :, self.prim_slice_230_starts::self.prim_slice_230_steps, :]
        opt_prim_slice_239 = opt_prim_slice_230[:, :, :, self.prim_slice_239_starts::self.prim_slice_239_steps]
        opt_rsub_246 = self.rsub_246_input_1 - opt_prim_slice_239 * self.rsub_246_alpha
        opt_mul_254 = opt_rsub_246 * self.mul_254_input_1
        visualbertencoder_0_opt = self.visualbertencoder_0(opt_concat_1139, opt_mul_254)
        return visualbertencoder_0_opt
