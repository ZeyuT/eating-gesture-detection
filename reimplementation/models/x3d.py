import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from reimplementation.models import fb_models

def generate_model(config_path, weights_path,network):
    model = fb_models.X3D(config_path)    
    ckpt = torch.load(weights_path)
    model.load_state_dict(ckpt['model_state'])
    if network == 'x3d-s':
        #x3d-s is used for frame-wise prediction
        model.head = X3DAlterHead(dim_in=192,
                                dim_inner=432,
                                dim_out=2048,
                                num_classes=3,
                                pool_size=[13, 5, 5],
                                dropout_rate=0.5,
                                task='loc')
    elif network == 'x3d-m':
        model.head = X3DAlterHead(dim_in=192,
                                dim_inner=432,
                                dim_out=2048,
                                num_classes=3,
                                pool_size=[16, 7, 7],
                                dropout_rate=0.5)
    elif network == 'x3d-l':
        model.head = X3DAlterHead(dim_in=192,
                                dim_inner=432,
                                dim_out=2048,
                                num_classes=3,
                                pool_size=[16, 10, 10],
                                dropout_rate=0.5)
        
    
    return model


class X3DAlterHead(nn.Module):
    """
    X3D head.
    This layer performs a fully-connected projection during training, when the
    input size is 1x1x1. It performs a convolutional projection during testing
    when the input size is larger than 1x1x1. If the inputs are from multiple
    different pathways, the inputs will be concatenated after pooling.
    """

    def __init__(
        self,
        dim_in,
        dim_inner,
        dim_out,
        num_classes,
        pool_size,
        dropout_rate=0.0,
        act_func="softmax",
        inplace_relu=True,
        eps=1e-5,
        bn_mmt=0.1,
        norm_module=nn.BatchNorm3d,
        bn_lin5_on=False,
        task='class'
    ):
        """
        The `__init__` method of any subclass should also contain these
            arguments.
        X3DHead takes a 5-dim feature tensor (BxCxTxHxW) as input.

        Args:
            dim_in (float): the channel dimension C of the input.
            num_classes (int): the channel dimensions of the output.
            pool_size (float): a single entry list of kernel size for
                spatiotemporal pooling for the TxHxW dimensions.
            dropout_rate (float): dropout rate. If equal to 0.0, perform no
                dropout.
            act_func (string): activation function to use. 'softmax': applies
                softmax on the output. 'sigmoid': applies sigmoid on the output.
            inplace_relu (bool): if True, calculate the relu on the original
                input without allocating new memory.
            eps (float): epsilon for batch norm.
            bn_mmt (float): momentum for batch norm. Noted that BN momentum in
                PyTorch = 1 - BN momentum in Caffe2.
            norm_module (nn.Module): nn.Module for the normalization layer. The
                default is nn.BatchNorm3d.
            bn_lin5_on (bool): if True, perform normalization on the features
                before the classifier.
        """
        super(X3DAlterHead, self).__init__()
        self.pool_size = pool_size
        self.dropout_rate = dropout_rate
        self.num_classes = num_classes
        self.act_func = act_func
        self.eps = eps
        self.bn_mmt = bn_mmt
        self.inplace_relu = inplace_relu
        self.bn_lin5_on = bn_lin5_on
        self.task = task

        self._construct_head(dim_in, dim_inner, dim_out, norm_module)
        
    def _construct_head(self, dim_in, dim_inner, dim_out, norm_module):

        self.conv_5 = nn.Conv3d(
            dim_in,
            dim_inner,
            kernel_size=(1, 1, 1),
            stride=(1, 1, 1),
            padding=(0, 0, 0),
            bias=False,
        )
        self.conv_5_bn = norm_module(
            num_features=dim_inner, eps=self.eps, momentum=self.bn_mmt
        )
        self.conv_5_relu = nn.ReLU(self.inplace_relu)

        if self.pool_size is None:
            self.avg_pool = nn.AvgPool3d((1, 1, 1))
        else:
            if self.task == 'class':
                self.avg_pool = nn.AvgPool3d(self.pool_size, stride=1)
            elif self.task == 'loc':
                self.avg_pool = nn.AvgPool3d((1,self.pool_size[1],self.pool_size[2]), stride=1)
                
        self.lin_5 = nn.Conv3d(
            dim_inner,
            dim_out,
            kernel_size=(1, 1, 1),
            stride=(1, 1, 1),
            padding=(0, 0, 0),
            bias=False,
        )
        if self.bn_lin5_on:
            self.lin_5_bn = norm_module(
                num_features=dim_out, eps=self.eps, momentum=self.bn_mmt
            )
        self.lin_5_relu = nn.ReLU(self.inplace_relu)

        if self.dropout_rate > 0.0:
            self.dropout = nn.Dropout(self.dropout_rate)
        # Perform FC in a fully convolutional manner. The FC layer will be
        # initialized with a different std comparing to convolutional layers.
        self.projection = nn.Linear(dim_out, self.num_classes, bias=True)

        # Softmax for evaluation and testing.
        if self.act_func == "softmax":
            self.act = nn.Softmax(dim=4)
        elif self.act_func == "sigmoid":
            self.act = nn.Sigmoid()
        else:
            raise NotImplementedError(
                "{} is not supported as an activation"
                "function.".format(self.act_func)
            )

    def forward(self, inputs):
        # In its current design the X3D head is only useable for a single
        # pathway input.
        assert len(inputs) == 1, "Input tensor does not contain 1 pathway"
        x = self.conv_5(inputs[0])
        x = self.conv_5_bn(x)
        x = self.conv_5_relu(x)
        x = self.avg_pool(x)

        x = self.lin_5(x)
        if self.bn_lin5_on:
            x = self.lin_5_bn(x)
        x = self.lin_5_relu(x)

        # (N, C, T, H, W) -> (N, T, H, W, C).
        x = x.permute((0, 2, 3, 4, 1))
        # Perform dropout.
        if hasattr(self, "dropout"):
            x = self.dropout(x)
        x = self.projection(x)
        
        if self.task == 'class':
            x = self.act(x)
            x = x.view(x.shape[0], -1)
            return x
        elif self.task == 'loc':
            fc_output = x 
            x = self.act(x)
            x = x.view(x.shape[0],x.shape[1], -1)
            fc_output = fc_output.view(fc_output.shape[0],fc_output.shape[1], -1)
            return x,fc_output

