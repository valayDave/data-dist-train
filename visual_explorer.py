import os 
from os import listdir
import glob
from typing import List
import pandas
import seaborn
import torch
from distributed_trainer import MODEL_FILENAME,MODEL_META_FILENAME,ExperimentBundle,ModelBundle,ExperimentResultsBundle,ConfusionMatrix,DATEFORMAT
import streamlit as st
import plotly.figure_factory as ff
from datetime import datetime
import plotly.graph_objects as go
from fraud_network import FraudFFNetwork,FraudCNNModel
from plotly.subplots import make_subplots


# MODEL_ROOTPATH = 'remote-model'
MODEL_ROOTPATH = 'model_data/fraud_model'
DEFAULT_TRAINER = 'FraudDistributedTrainer'

AVAILABLE_TRAINERS = [
    {'trainer':'FraudDistributedTrainer','is_distributed':True},
    {'trainer':'FraudTrainer','is_distributed':False}
]

VIEW_OPTIONS = [
    'Results View',
    'Data View'
]

def get_experiments(model_root=MODEL_ROOTPATH,trainer=DEFAULT_TRAINER,is_distributed=True,load_model=False,load_meta=True):
    collected_experiments = []
    experiments = listdir(os.path.join(model_root,trainer))
    default_rank = 'Rank-0'
    for experiment in experiments:
        load_path = None
        return_data = {
            'model' : None,
            'meta': None,
            'path': None,
        }
        if is_distributed:
            experiment_path = os.path.join(model_root,trainer,experiment,default_rank)
        else:
            experiment_path = os.path.join(model_root,trainer,experiment)
        checkpoint = max(listdir(experiment_path))
        load_path = os.path.join(experiment_path,checkpoint,'**','*.pt')
        
        for file in glob.glob(load_path, recursive = True):
            if MODEL_FILENAME in file:
                if load_model:
                    return_data['model'] = torch.load(file)
            if MODEL_META_FILENAME in file:
                if load_meta:
                    return_data['meta'] = ExperimentBundle(**torch.load(file))
        
        return_data['path'] = load_path
        collected_experiments.append(return_data)
    return collected_experiments

def get_experiment(
        load_path,
        load_model=False,\
        load_meta=True,
    ):
    return_data = {
        'model' : None,
        'meta': None,
        'path': None,
    }
    for file in glob.glob(load_path, recursive = True):
        if MODEL_FILENAME in file:
            if load_model:
                return_data['model'] = torch.load(file, map_location=torch.device('cpu'))
        if MODEL_META_FILENAME in file:
            if load_meta:
                return_data['meta'] = ExperimentBundle(**torch.load(file))
    
    return_data['path'] = load_path
    return return_data
    


class DataView:
    def __init__(self):
        st.markdown('# Fraud Data Exploration Dashboard')
        st.sidebar.title('Experiment Filter Options')
        selected_trainer = st.sidebar.selectbox(
            'Select A Trainer from The List',
            AVAILABLE_TRAINERS,
            0,
            lambda x:x['trainer']
        )
        show_all_loss = st.sidebar.checkbox('Show ALL Exp Losses In One Plot')
        model_experiment_meta = get_experiments(**selected_trainer)

        meta = list(map(lambda x:x['meta'],model_experiment_meta))

        self.build_sidebar_meta(meta)
        if show_all_loss:
            self.build_overall_exp(meta)
        selected_experiment = st.sidebar.selectbox('Select Experiment For %s'%selected_trainer['trainer'],model_experiment_meta,0,lambda x : x['meta'].created_on)
        selected_model = self.get_model_bundle(selected_experiment['path'])
        
        self.display_experiment(selected_experiment['meta'],selected_model)

    def get_model_bundle(self,file_path):
        dat = get_experiment(file_path,load_model=True,load_meta=False)
        model_bundle = ModelBundle(**dat['model'])
        return model_bundle

    def build_sidebar_meta(self,model_experiment_meta:List[ExperimentBundle]):
        md_head = '''
        #### Number Of Conducted Experiments : {len_exp}\n

        #### Core Stats
        '''.format(len_exp=str(len(model_experiment_meta)))
        st.sidebar.markdown(md_head)
        df = pandas.DataFrame(pandas.json_normalize([{'train_args':exp.train_args,'dataset_metadata':exp.dataset_metadata,'distributed':exp.distributed} for exp in model_experiment_meta]))

    def build_overall_exp(self,model_experiment_meta:List[ExperimentBundle]):
        loss_fig = go.Figure()
        loss_fig.update_layout(title='Validation Losses of Different Models')
        for bundle in model_experiment_meta:
            epoch_results = []
            validation_results_df = pandas.DataFrame(bundle.validation_epoch_results)
            validation_results_df = validation_results_df[['epoch','losses','created_on','accuracy','batch_time']]
            loss_fig.add_trace(
                go.Scatter(x=validation_results_df['epoch'],y=validation_results_df['losses'],name=bundle.created_on,line_shape='linear'),
            )
        st.plotly_chart(loss_fig)

    def show_losses(self,bundle:ExperimentBundle):
        loss_fig = go.Figure()
        loss_fig.update_layout(title='Validation Losses of Model : %s'%bundle.created_on)
        validation_results_df = pandas.DataFrame(bundle.validation_epoch_results)
        validation_results_df = validation_results_df[['epoch','losses']]
        loss_fig.add_trace(
            go.Scatter(x=validation_results_df['epoch'],y=validation_results_df['losses'],name=bundle.created_on,line_shape='linear'),
        )
        st.plotly_chart(loss_fig)
    
    def show_acc(self,bundle:ExperimentBundle):
        acc_fig = go.Figure()
        acc_fig.update_layout(title='Validation Accuracy of Model : %s'%bundle.created_on)
        validation_results_df = pandas.DataFrame(bundle.validation_epoch_results)
        validation_results_df = validation_results_df[['epoch','accuracy']]
        acc_fig.add_trace(
            go.Scatter(x=validation_results_df['epoch'],y=validation_results_df['accuracy'],name=bundle.created_on,line_shape='linear'),
        )
        st.plotly_chart(acc_fig)

    @staticmethod
    def get_experiment_meta_dict(bundle:ExperimentBundle,model:ModelBundle):
        last_ep_bundle = bundle.validation_epoch_results[-1]
        conf_mat = ConfusionMatrix(**last_ep_bundle['confusion_matrix'])
        ds_meta = bundle.dataset_metadata
        is_distributed = bundle.distributed
        # Time Extraction
        train_start_time = ExperimentResultsBundle(**bundle.train_epoch_results[0]).created_on
        train_end_time = ExperimentResultsBundle(**bundle.train_epoch_results[-1]).created_on
        start_date = datetime.strptime(train_start_time,DATEFORMAT)
        end_date = datetime.strptime(train_end_time,DATEFORMAT)
        time_taken = (end_date - start_date).total_seconds()
        # Score Extraction
        precision = round(conf_mat.precision_macro_average(),3)
        recall = round(conf_mat.recall_macro_average(),3)
        # Make One Object of all data and return it. 
        return dict(
                distributed = is_distributed,
                loss_fn = model.loss_fn,
                batch_size=bundle.train_args['batch_size'],
                num_epochs=bundle.train_args['num_epochs'],
                learning_rate=model.optimizer_args['lr'],
                model_name = model.model_name if model.model_name is not None else '*Model-Name-Not-Logged*',
                training_note = bundle.note if bundle.note is not None else '*No Training Note*',
                train_start_time=start_date,
                train_end_time = end_date,
                time_taken = time_taken,
                precision = precision,
                recall = recall
            )
        
    def display_experiment(self,bundle:ExperimentBundle,model:ModelBundle):
        last_ep_bundle = bundle.validation_epoch_results[-1]
        conf_mat = ConfusionMatrix(**last_ep_bundle['confusion_matrix'])
        ds_meta = bundle.dataset_metadata
        is_distributed = bundle.distributed

        model_meta_print = '''
        ## Model Metadata\n
        \n
        - Epochs : {num_epochs}\n
        - Loss Function : {loss_fn}\n
        - Batch Size : {batch_size}\n
        - Learning Rate : {learning_rate}\n
        - Distribtued Training : {distributed}\n
        - Model Name : {model_name}\n
        - Experiment Note : {training_note}\n
        - Train Start Time : {train_start_time}\n
        - Train End Time : {train_end_time}\n
        - Time Taken : {time_taken} Seconds\n
        - Precision : {precision}\n
        - Recall : {recall}\n
        '''.format(
            **self.get_experiment_meta_dict(bundle,model)
        )
        st.markdown(model_meta_print)
        
        if bundle.distributed:
            dataset_meta = '''
            ### Dataset Metadata For Distributed Experiment {exp_name}\n
            Sample Used : {is_sample}\n
            Uniform Label Distribution  : {uniform_label_distribution}\n
            Label Split Distribution : {label_dist}\n
            Test Set Size : {test_set_portion} %\n
            Selected Split : {selected_split}\n
            '''.format(
                **dict(
                    exp_name=bundle.created_on,
                    is_sample = ds_meta['sample'] is not None,
                    uniform_label_distribution=ds_meta['uniform_label_distribution'],
                    label_dist= ','.join([str(round(i*100,3))+' ' for i in ds_meta['label_split_values']]) if ds_meta['label_split_values'] else 'None',
                    test_set_portion=str(ds_meta['test_set_portion']*100),
                    selected_split = '*Not Selected*' if 'selected_split' not in ds_meta else ds_meta['selected_split']
                )
            )
            st.markdown(dataset_meta)
        self.show_losses(bundle)
        self.show_acc(bundle)
        self.print_conf_matrix(conf_mat)

    def print_conf_matrix(self,conf_mat:ConfusionMatrix,set_val='Test Set'):
        df = pandas.DataFrame(conf_mat.conf_mat, range(len(conf_mat.conf_mat)), range(len(conf_mat.conf_mat)))
        st.markdown('#### Confusion Matrix Of %s'%set_val)
        seaborn.heatmap(df, annot=True,linewidths=.5,fmt='.2f')
        st.pyplot()


class ResultsView:
    pass

def run_app():
    # st.sidebar.multiselect('View Options',)
    DataView()

if __name__=="__main__":
    run_app()