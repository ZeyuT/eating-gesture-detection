import torch
from models import cnn_lstm,slowfast,x3d

def get_model(network):
    """CNN-LSTM category
    """
    if network in ["lstm-r34","lstm-r50","lstm-r101"]:
        model = cnn_lstm.generate_model(seq_len=16,
                                        network=network)
        model_type = "seq2seq"
        inference_type = "seq2one"
        fps = 8 #in hz
        seq_len = 16

    """SlowFast category
    """
    if network == "slowfast-r50":
        model = slowfast.generate_model(
                    config_path="../reimplementation/models/config/SLOWFAST_8x8_R50_stepwise.yaml",
                    weights_path="../reimplementation/pre_trained/SLOWFAST_8x8_R50_stepwise.pkl",
                    network=network) 
        model_type = "seq2one"
        inference_type = "seq2one"
        fps = 16 #in hz
        seq_len = 32
    """X3D category
    """        
    if network in ["x3d-s","x3d-m","x3d-l"]:
        if network == "x3d-s":
            # TODO: Generate frame dataset for the fps, H and W
            config_path = "../reimplementation/models/config/X3D_S.yaml"
            weight_path = "../reimplementation/pre_trained/x3d_s.pyth"
            fps = 6 #in hz
            seq_len = 16
        elif network == "x3d-m":
            # TODO: Generate frame dataset for the fps, H and W
            config_path = "../reimplementation/models/config/X3D_M.yaml"
            weight_path = "../reimplementation/pre_trained/x3d_m.pyth"
            fps = 5 #in hz
            seq_len = 13
        elif network == "x3d-l":
            config_path = "../reimplementation/models/config/X3D_L.yaml"
            weight_path = "../reimplementation/pre_trained/x3d_l.pyth"   
            fps = 6 #in hz
            seq_len = 16
        model = x3d.generate_model(config_path=config_path,
                                    weights_path=weight_path,
                                    network=network)     
        model_type = "seq2one"
        inference_type = "seq2one"


    if not model:
        raise ImportError("Can not load model")
    return model, model_type, inference_type, fps, seq_len
