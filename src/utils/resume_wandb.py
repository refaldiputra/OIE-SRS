import yaml
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-p','--path', type=str,help='path to dir that has wandb.yaml', default='/home/jovyan/shared/refaldi/llm_id_recsys/configs/logger/')
parser.add_argument('-i','--id', type=str)
parser.add_argument('-r','--resume', type=str, default='must')

args = parser.parse_args()

istream_path = args.path + 'wandb.yaml'
outstream_path = args.path + 'wandb_resume.yaml'
with open(istream_path) as istream:
    y = yaml.safe_load(istream)
    y['wandb']['id'] = args.id
    y['wandb']['resume'] = args.resume

with open(outstream_path, 'w') as outstream:
    yaml.dump(y, outstream, default_flow_style=False, sort_keys=False)

#print(args.path, args.id, args.resume)