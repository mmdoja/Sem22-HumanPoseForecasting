# from ray import tune
HYPERPARAM_CONFIGS = {
    "StateSpace2dJoints":{
        # "lr": tune.loguniform(1e-4, 1e-1),
        # "encoder_layers": [tune.choice([32, 64])],
        # "n_cells": tune.choice([1, 2]),
        # "rnn_dim": tune.choice([32, 64, 128, 256]),
        # "batch_size": tune.choice([32, 64]),

        "method": "random",
        "metric": {
            "name": "val_PCK",
            "goal": "maximize"
        },
        "parameters": {
            "lr": {
                "distribution": "log_uniform_values",
                "min": 1e-5,
                "max": 1e-2
            },
            "encoder_layers":{
                "values": [32, 64, 128]
            },
            "n_cells":{
                "values": [1, 2]
            },
            "rnn_dim":{
                "values": [32, 64, 128, 256]
            },
            "batch_size":{
                "values": [32, 64, 128]
            },
            "cell_type": {
                "values": ["LSTM", "GRU"]
            },
            "loss":{
                "values": ["MSELoss", "L1Loss"]
            },
        },
        "early_terminate": {
            "type": "hyperband",
            "min_iter": 3

        }       
    },
    "Autoregressive2dJoints":{

        "method": "random",
        "metric": {
            "name": "val_PCK",
            "goal": "maximize"
        },
        "parameters": {
            "lr": {
                "distribution": "log_uniform_values",
                "min": 1e-5,
                "max": 1e-2
            },
            "encoder_layers":{
                "values": [32, 64, 128]
            },
            "n_cells":{
                "values": [1, 2]
            },
            "rnn_dim":{
                "values": [32, 64, 128, 256]
            },
            "batch_size": {
                "values": [32, 64, 128]
            },
            "teacher_forcing_ratio":{
                "values": [0, 0.5]
            },
            "cell_type": {
                "values": ["LSTM", "GRU"]
            },
            "loss":{
                "values": ["MSELoss", "L1Loss"]
            },
        },
        "early_terminate": {
            "type": "hyperband",
            "min_iter": 3

        }       
    },
    "StateSpaceHeatmaps":{

        "method": "random",
        "metric": {
            "name": "val_PCK",
            "goal": "maximize"
        },
        "parameters": {
            "lr": {
                "distribution": "log_uniform_values",
                "min": 1e-5,
                "max": 1e-2
            },
            "enc_layers_ch":{
                "values": [32, 64]
            },
            "n_cells":{
                "values": [1, 2]
            },
            "rnn_ch":{
                "values": [32, 64, 128]
            },
            "batch_size": {
                "values": [32, 64]
            },
            "loss":{
                "values": ["MSELoss", "SSIMLoss"]
            },
        },
        "early_terminate": {
            "type": "hyperband",
            "min_iter": 3

        }       
    },
    "AutoregressiveHeatmaps":{

        "method": "random",
        "metric": {
            "name": "val_PCK",
            "goal": "maximize"
        },
        "parameters": {
            "lr": {
                "distribution": "log_uniform_values",
                "min": 1e-5,
                "max": 1e-2
            },
            "enc_layers_ch":{
                "values": [32, 64]
            },
            "n_cells":{
                "values": [1, 2]
            },
            "rnn_ch":{
                "values": [32, 64, 128]
            },
            "batch_size": {
                "values": [32, 64]
            },
            "teacher_forcing_ratio":{
                "values": [0, 0.5]
            },
            "loss":{
                "values": ["MSELoss", "SSIMLoss"]
            },
        },
        "early_terminate": {
            "type": "hyperband",
            "min_iter": 3

        }       
    },
}