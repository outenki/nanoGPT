# train a miniature character-level shakespeare model
# good for debugging and playing on macbooks and such

out_dir = 'data/wikitext_100k/12-12-768'  # <-- change to your output directory
eval_interval = 250  # keep frequent because we'll overfit
eval_iters = 200
log_interval = 10  # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = False  # override via command line if you like
wandb_project = 'wikitext_100k'  # <-- change to your project name
wandb_run_name = 'mini-gpt'

dataset = 'wikitext_100k'  # <-- change to your dataset name
gradient_accumulation_steps = 1
batch_size = 64
block_size = 256  # context of up to 256 previous characters

# baby GPT model :)
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0
