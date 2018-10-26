# dataset
--data=chunking

# inference type
--inf=Variational

# kernel
--cov=SquaredExponential

# length scale
--length_scale=10.0

# Graph mode
--tf_mode=graph

# No plotting
--plot=

# save prediction
--preds_path=./results

# training parameters
--lr=0.002
--optimizer=RMSPropOptimizer
--lr_drop_steps=50
--lr_drop_factor=0.1
--train_steps=1000
--chkpnt_steps=500
--loo_steps=50
--batch_size=2248
--eval_epochs=10000

# save the trained model
--save_dir=./results
--model_name=chunking1
