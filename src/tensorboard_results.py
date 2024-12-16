from tensorboard import notebook

log_dir = "peft_training"
notebook.start("--logdir {}".format(log_dir))
