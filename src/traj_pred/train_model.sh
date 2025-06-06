# To train a new model
torchrun --nproc_per_node=1 --master_port=29500 src/traj_pred/train.py \
        --eval_every=1 \
        --vis_every=1 \
        --batch_size=256 \
        --eval_batch_size=256 \
        --preprocess_workers=16 \
        --log_dir=./data/trained_models/trajectory_prediction \
        --train_epochs=1 \
        --conf=./config/pedestrians.json \
        --trajdata_cache_dir=./data/pedestrian_datasets/.unified_data_cache \
        --history_sec=2.8 --prediction_sec=4.8 \
        --K=1 \
        --alpha_init=1.54e-05 \
        --augment_input_noise=0.57 \
        --grad_clip=0.909 \
        --learning_rate=0.016 \
        --sigma_eps_init=0.0002 \
        --contrastive_weight=50.0 \
        --train_data=eupeds_eth-train \
        --eval_data=eupeds_eth-val \
        --log_tag=eth

# To update a pre-trained model
torchrun --nproc_per_node=1 --master_port=29500 train.py \
        --eval_every=1 \
        --vis_every=1 \
        --batch_size=256 \
        --eval_batch_size=256 \
        --preprocess_workers=16 \
        --log_dir=./data/trained_models/trajectory_prediction \
        --load_dir=./data/trained_models/trajectory_prediction \
        --train_epochs=15 \
        --conf=./config/pedestrians.json \
        --trajdata_cache_dir=./data/pedestrian_datasets/.unified_data_cache \
        --train_data=eth-train \
        --eval_data=eth-val \
        --history_sec=2.8 --prediction_sec=4.8 \
        --K=1 \
        --alpha_init=1.54e-05 \
        --augment_input_noise=0.57 \
        --grad_clip=0.909 \
        --learning_rate=0.016 \
        --sigma_eps_init=0.0002 \
        --contrastive_weight=50.0