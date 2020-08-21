from metaflow import FlowSpec,step,Parameter
import os
            
class DistTrainerFlow(FlowSpec):

    note = Parameter('Note',
                      help='Note To Leave For Training',
                      default="No NOTE")
    
    backend = Parameter('backend',help='Training Backend',default='gloo')

    @step
    def start(self):
        from parameter_map import Parameter,ParameterMap,ScriptInput
        param_list_dist = [
            Parameter(
                name='block_size',
                values=[2,10,100,500,1000]
            ),
            Parameter(
                name='approach',
                values=['random','round_robin'],
                quoted=True
            ),
            Parameter(
                name='num_workers',
                values=[2,3,4,5,6,7,8],
            )
        ]
        st = ScriptInput(
            base_command = '',
            param_list = param_list_dist
        )
        arg_list_dist = [
            Parameter(
                name='batch_size',
                # values=[2**(i+1) for i in range(7)] # [256, 512, 1024, 2048, 4096] 
                values = [8,16,32,64]
            ),
            Parameter(
                name='epochs',
                values=[100] 
            ),
            Parameter(
                name='backend',
                values=[self.backend] 
            ),
            Parameter(
                name='master_ip',
                values=['127.0.0.1'] 
            ),
            Parameter(
                name='master_port',
                random_value=True, # Port RANDOMIZATION COULD BE REALLY USEFUL
                values=['12355'] 
            ),
            Parameter(
                name='learning_rate',
                values=[0.1,0.01]
            ),
            Parameter(
                name='model',
                values=['ResNet50'] 
            ),
            
            Parameter(
                name='note',
                values=[self.note],
                quoted=True
            ),
        ]
        pp = ParameterMap(st)
        arg_mp = ParameterMap(ScriptInput(
            base_command='',
            param_list = arg_list_dist
        ))
        import random 

        # self.dispatcher_params = random.sample(pp.object_dicts,1)
        self.dispatcher_params = pp.object_dicts
        self.all_model_args = arg_mp.object_dicts #random.sample(arg_mp.object_dicts,2)
        
        self.next(self.create_dataset,foreach='dispatcher_params')
    
    @step
    def create_dataset(self):
        from distributed_trainer.data_dispatcher import DispatcherControlParams
        import cifar_dataset
        dispatcher_params = DispatcherControlParams(
            **self.input
        )
        self.dispatch_params = self.input
        distributed_dataset,datastore = cifar_dataset.get_distributed_dataset(dispatcher_params)
        self.dist_datastore = datastore
        self.distributed_dataset = distributed_dataset

        self.next(self.train,foreach='all_model_args')

    @step
    def train(self):
        from distributed_cifar import run_trainer_from_dataset
        self.trainer_args = self.input
        experiment_bundle, model_bundle = run_trainer_from_dataset(
            **self.trainer_args,
            world_size = len(self.distributed_dataset.blocks),
            distributed_dataset = self.distributed_dataset
        )
        self.experiment_bundle = experiment_bundle
        self.model_bundle = model_bundle
        self.next(self.join_trainer)

    @step
    def join_trainer(self,inputs):
        self.data_bundles = [ # $ List[Tuple[dispatch_param, trainer_arg, dist_datastore]]

        ]
        self.model_bunles = [ # $ List[Tuple[dispatch_param, trainer_arg, model_bundle]]

        ]
        self.results_bundles = [ # $ List[Tuple[dispatch_param, trainer_arg, experiment_bundle]]

        ]
        for collected_data in inputs:
            self.data_bundles.append(
                (
                    collected_data.dispatch_params,
                    collected_data.trainer_args,
                    collected_data.dist_datastore
                )
            )
            self.model_bunles.append(
                (
                    collected_data.dispatch_params,
                    collected_data.trainer_args,
                    collected_data.model_bundle

                )
            )
            self.results_bundles.append(
                (
                    collected_data.dispatch_params,
                    collected_data.trainer_args,
                    collected_data.experiment_bundle
                )
            )
        self.next(self.join_disp)
    
    @step
    def join_disp(self,inputs):
        self.data_bundles = [

        ]
        self.model_bundles = [

        ]
        self.results_bundles = [

        ]
        for collected_data in inputs:
            self.data_bundles.extend(collected_data.data_bundles)
            self.results_bundles.extend(collected_data.results_bundles)
            self.model_bundles.extend(collected_data.model_bunles)
        self.next(self.end)
    
    @step
    def end(self):
        print("Done Flow")

if __name__=='__main__':
    DistTrainerFlow()