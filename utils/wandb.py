import wandb
import numpy as np
import torch

class WandbManager:
    def __init__(self, project, run_name, config, run_id=None):
        self.config = config
        self.run_id = run_id

        self.run = wandb.init(
            project=project, 
            name=run_name, 
            config=self.config,
            id=self.run_id,
            resume="allow" if self.run_id else None,
        )
    
    def get_run(self):
        return self.run
        
    def log_torch_model(self, name, path, aliases, config):
        artifact = wandb.Artifact(name=name, type='model', metadata=dict(config))
        artifact.add_file(path)
        self.run.log_artifact(artifact, aliases=aliases)

    def log_everything(self, run_name, path):
        artifact = wandb.Artifact(name=run_name, type='everything')
        artifact.add_dir(path)
        self.run.log_artifact(artifact)

    def log_dataframe(self, name, df):
        table = wandb.Table(dataframe=df)
        self.run.log({name: table})

    
class WandbTable:
    """
    Example:
        # Create a new WandbTable
        my_table = WandbTable()

        # Add data to the table
        my_table.add_data({"column1": "value1", "column2": "value2", "column3": "value3"})
        my_table.add_data({"column1": "value4", "column2": "value5", "column4": "value6"})

        # Log the table
        run.log({"my-table": my_table.get_table()})
    """
    def __init__(self):
        self.data = []
        self.columns = set()
        self.table = None

    def add_data(self, data_dict):
        self.data.append(data_dict)
        self.columns.update(data_dict.keys())

    def get_table(self):
        self.table = wandb.Table(columns=list(self.columns))
        for data_dict in self.data:
            row = [data_dict.get(col, None) for col in self.columns]
            self.table.add_data(*row)
        return self.table


def wandb_image(path):
    img = wandb.Image(path)
    return img


def download_everything(entity, project, name):
    api = wandb.Api()
    artifact = api.artifact(f'{entity}/{project}/{name}:latest', type='everything')
    artifact.download(f'runs/{name}')



def wandb_make_batch_grid(gt, pred, step_count):
    """
    gt, pred: [B, C, H, W] tensors in [-1, 1] or [0, 1]
    returns: wandb.Image with shape [2H, B*W, C]
    """

    # map to [0, 1]
    gt = (gt.clamp(-1, 1) + 1) / 2
    pred = (pred.clamp(-1, 1) + 1) / 2

    # convert to HWC
    gt = gt.permute(0, 2, 3, 1)
    pred = pred.permute(0, 2, 3, 1)

    # concat GT horizontally
    gt_row = torch.cat(list(gt), dim=1)     # [H, B*W, C]
    pred_row = torch.cat(list(pred), dim=1) # [H, B*W, C]

    # stack GT on top of GEN
    grid = torch.cat([gt_row, pred_row], dim=0)  # [2H, B*W, C]

    return wandb.Image(
        grid.cpu().numpy(),
        caption=f"GT (top) | Generated (bottom) — step {step_count}"
    )