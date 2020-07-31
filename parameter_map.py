from dataclasses import dataclass,field
from dataclasses import asdict as dataclass_to_dict
from typing import List
import itertools
import random

# https://discuss.pytorch.org/t/multiprocessing-failed-with-torch-distributed-launch-module/33056/21

@dataclass
class Parameter:
    name:str
    values:List = field(default_factory=lambda:[])
    quoted:bool=False
    random_value:bool=False

    def as_arg(self):
        return '--'+self.name
    
    def parse_value(self,value):
        if self.random_value:
            value = random.randint(10000,40000)
        if self.quoted:
            value = "\""+value+"\""
        return value
    

@dataclass
class ScriptInput:
    param_list:List[Parameter]
    base_command:str

SPACE = ' '
TAB = '\t'
NEWLINE= '\n'
ARG_SEPERATOR = SPACE+'\\'+NEWLINE+TAB


class ParameterMap:
    def __init__(self,script_input:ScriptInput):
        if type(script_input) != ScriptInput:
            raise("Required List[Parameter] As Strict Input")
        self.static_params = []
        self.multi_valueparams = []


        for param in script_input.param_list:
            if len(param.values) == 1:
                self.static_params.append(param)
            else:
                self.multi_valueparams.append(param)
        
        # List[List[Params:onevalue]]
        param_bins = [

        ]
        for param in self.multi_valueparams:
            curr_bin = []
            for value in param.values:
                curr_bin.append(
                    Parameter(
                        name=param.name,
                        quoted=param.quoted,
                        values=[value]
                    )
                )
            param_bins.append(curr_bin)
        
        self.parameter_combination = self.get_key_map(param_bins)

        self.command_list = []
        for combination in self.parameter_combination:
            static_command = script_input.base_command + SPACE + ' '.join([
                                ARG_SEPERATOR.join([self.construct_arg(arg,arg.values[0]) for arg in self.static_params+list(combination)])
                                ])
            self.command_list.append(static_command)
        
    def to_shell_file(self,file_path):
        head_data = 'echo "Training Session Starting With About %d Jobs"\n'%len(self.command_list)
        notification_data = '\n\necho "STARTING NEW EXECUTION OF CODE: $(date)"\n\n'
        print_string = notification_data.join(
            self.command_list
        )
        with open(file_path,'w+') as f: 
            f.write(head_data+notification_data+print_string)

        
    @staticmethod
    def get_key_map(arr):
        finalmap = []
        for i in itertools.product(*arr):
            finalmap.append(i)
        return finalmap


    @staticmethod
    def construct_arg(arg:Parameter,value):
        return arg.as_arg()+SPACE+str(arg.parse_value(value))


if __name__ == "__main__":
    param_list_dist = [
        Parameter(
            name='batch_size',
            values=[2**(i+7) for i in range(7)] # [128, 256, 512, 1024, 2048, 4096] 
        ),
        Parameter(
            name='epochs',
            values=[30] 
        ),
        Parameter(
            name='backend',
            values=['nccl'] 
        ),
        Parameter(
            name='master_ip',
            values=['127.0.0.1'] 
        ),
        Parameter(
            name='master_port',
            random_value=True,
            values=['12355'] 
        ),
        Parameter(
            name='learning_rate',
            values=[0.0001,0.001]
        ),
        Parameter(
            name='model',
            values=['CNN','FF'] 
        ),
        Parameter(
            name='use_split',
            values=['n_5_b_2','n_5_b_90','n_5_b_110','n_5_b_130']
        ),
        Parameter(
            name='note',
            values=['Testing Run Of Auto Parameterization'],
            quoted=True
        ),
    ]
    st = ScriptInput(
        base_command = 'python distributed_cred.py distributed',
        param_list = param_list_dist
    )

    pp = ParameterMap(st)
    pp.to_shell_file('./command_list.sh')