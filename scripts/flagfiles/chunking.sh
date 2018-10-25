# dataset
--data=chunking

# inference type
--inf=Variational

# kernel
--cov=SquaredExponential

# Graph mode
--tf_mode=graph

# No plotting
--plot=

# save prediction
--preds_path=./results

# training parameters
--lr=0.05
--optimizer=RMSPropOptimizer
--lr_drop_steps=50
--lr_drop_factor=0.1
--train_steps=500

save the trained model
--save_dir=./result
--model_name=chunking1
