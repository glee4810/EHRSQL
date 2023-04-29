import argparse

class Config():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialize_parser()

    def add_optim_param(self):
        self.parser.add_argument('--warmup_steps', type=int, default=0)
        self.parser.add_argument('--total_epoch', type=int, default=-1)
        self.parser.add_argument('--total_step', type=int, default=100000)
        self.parser.add_argument('--train_batch_size', type=int, default=4)
        self.parser.add_argument('--accumulation_steps', type=int, default=8)
        self.parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
        self.parser.add_argument('--scheduler_steps', type=int, default=None, 
                        help='total number of step for the scheduler, if None then scheduler_total_step = total_step')
        self.parser.add_argument('--optim', type=str, default='adam')
        self.parser.add_argument('--scheduler', type=str, default='fixed')
        self.parser.add_argument('--max_grad_norm', type=str, default=1.0)
        self.parser.add_argument('--weight_decay', type=float, default=0.1)
        self.parser.add_argument('--init_weights', type=bool, default=False, help='do not use pre-trained weights')

    def add_model_param(self):
        self.parser.add_argument('--dataset', type=str, help='dataset')
        self.parser.add_argument('--db_id', type=str, help='db_id', choices=['mimic3', 'eicu'])
        self.parser.add_argument('--train_data_path', type=str, help='train data path')
        self.parser.add_argument('--valid_data_path', type=str, help='eval data path')
        self.parser.add_argument('--output_dir', type=str, default='outputs', help='output directory')
        self.parser.add_argument('--output_file', type=str, default='prediction_raw.json', help='output file name')        
        self.parser.add_argument('--model_name', type=str, default='t5-base')
        self.parser.add_argument('--db_path', type=str, default=None)
        self.parser.add_argument('--add_schema', type=bool, default=False)
        self.parser.add_argument('--add_column_type', type=bool, default=False)
        self.parser.add_argument('--shuffle_schema', type=bool, default=False)
        self.parser.add_argument('--tables_path', type=str, default=None)
        self.parser.add_argument('--condition_value', type=bool, default=True)

    def initialize_parser(self):
        # basic parameters
        self.parser.add_argument('--exp_name', type=str, default=None, help='name of the experiment')
        self.parser.add_argument('--load_model_path', type=str, default=None, help='path for retraining')
        self.parser.add_argument("--device", type=str, default='cuda')
        self.parser.add_argument("--num_workers", type=int, default=50)
        
        # training parameters
        self.parser.add_argument('--random_seed', type=int, default=0)
        self.parser.add_argument('--report_every_step', type=int, default=50)
        self.parser.add_argument('--eval_batch_size', type=int, default=4)
        self.parser.add_argument('--save_every_step', type=int, default=-1)
        self.parser.add_argument('--save_every_epoch', type=bool, default=False)
        self.parser.add_argument('--show_eval_sample', type=bool, default=True)
        self.parser.add_argument('--eval_every_step', type=int, default=5000)
        self.parser.add_argument('--eval_metric', type=str, default='loss', choices=['loss', 'esm'])
        self.parser.add_argument('--keep_last_ckpt', type=int, default=-1)
        self.parser.add_argument('--early_stop_patience', type=int, default=-1)
        self.parser.add_argument('--training_data_ratio', type=float, default=1.0)
        self.parser.add_argument('--bf16', type=bool, default=False)

        # wandb parameters
        self.parser.add_argument('--use_wandb', type=bool, default=False)
        self.parser.add_argument('--wandb_project', type=bool, default=None)

    def add_eval_param(self):
        self.parser.add_argument('--num_beams', type=int, default=5)
        self.parser.add_argument('--max_length', type=int, default=512)
        self.parser.add_argument('--repetition_penalty', type=float, default=1.0)
        self.parser.add_argument('--length_penalty', type=float, default=1.0)
        self.parser.add_argument('--early_stopping', type=bool, default=True)
        self.parser.add_argument('--num_samples', type=int, default=1)

    def get_param(self, use_model_param=False,
                        use_optim_param=False,
                        use_eval_param=False):
        if use_model_param:
            self.add_model_param()
        if use_optim_param:
            self.add_optim_param()
        if use_eval_param:
            self.add_eval_param()

    def parse(self):
        args = self.parser.parse_args()
        return args