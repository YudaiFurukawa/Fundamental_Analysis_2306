# Import comet_ml at the top of your file
from comet_ml import Experiment



#####################################
import pandas as pd
import numpy as np
import os, sys,random
from collections import deque
from sklearn.metrics import r2_score,mean_squared_error
from scipy.stats.stats import pearsonr
from model import Model_skip_211229 as net
import re, csv
import itertools

import utils



### Torch
import torch
from torch.utils import data
from torch.utils.data.dataset import Dataset
import torch.nn.functional as F
import torch.optim as optim
import time
# https://qiita.com/sugulu_Ogawa_ISID/items/62f5f7adee083d96a587
#  TODO test 
# Increase batch size or decrease # of parameters?
random.seed(7)
params = {'batch_size': 128,
          'shuffle': True,
          'num_workers': 8,
          'pin_memory':True}
val_params = {'batch_size': 128,
              'shuffle': True,
              'num_workers': 8,
              'drop_last':True,
              'pin_memory':True}


LEARNING_RATE = 1e-3
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print("using device...", device)

dir = "C:/Users/yften/OneDrive/Investing/02_final_data/"
# load_dir = '20200403_128_64_16_real'
# conn = sqlite3.connect("C:/Users/yften/OneDrive/Investing/db/inv_database.db")
# conn = sqlite3.connect("./inv_database.db")
# cur = conn.cursor()
# This is the data I am using
# Removed Minorities


### Input Data ###
data_dir = './data/'
id = '220423'
df_target = pd.read_csv(data_dir + 'all_tickers_returns_rank_'+id+'.csv').fillna(1e-6) ### y
# df_target = pd.read_csv(data_dir + 'alphas_'+id+'.csv').fillna(1e-6) ### y

df = pd.read_csv(data_dir + 'normalized_data_'+id+'.csv.gz') ### x
### 'Share_Price','Ticker','date' are needed for later
df = df[['Share_Price','Ticker','date','Cash_From_Operating_Activities_1y','Cash_From_Operating_Activities_5y'\
        ,'Revenues_1y','Revenues_5y','Total_Assets_1y','Total_Assets_5y','Current_Ratio_1y','Current_Ratio_5y','cus_profitMargin_1y'\
            ,'cus_profitMargin_5y','cus_ROE_1y','cus_ROE_5y','cus_Net_Profit_per_price','cus_Net_Profit_per_share_1y','cus_Net_Profit_per_share_5y'\
                ,'Cash_From_Operating_Activities','Cash_From_Financing_Activities','Cash_From_Investing_Activities','Equity_Before_Minorities'\
                    ,'Current_Ratio','Net_Profit','Revenues','Total_Assets','cus_ROE','cus_profitMargin','cus_Leverage','CPI_ALL'\
                        ,'NONFARM','M1SL','FED_FUND_RATE','B50001','VIX','TNX','cus_Cash_From_Financing_Activities_per_price'\
                            ,'cus_Cash_From_Investing_Activities_per_price','cus_Cash_From_Operating_Activities_per_price'\
                                ,'cus_Current_Assets_per_price','cus_Equity_Before_Minorities_per_price','cus_Net_Change_in_PPE_Intangibles_per_price'\
                                    ,'100001','100002','100003','100004','100005','100006','100007','100008','100009','100010','100011','100012','100013'\
                                        ,'101001','101002','101003','101004','101005','102001','102002','102003','102004','102005','102006','103001','103002',\
                                            '103003','103004','103005','103011','103013','103015','103018','103020','103026','104001','104002','104003',\
                                                '104004','104005','104006','104007','104013','105001','105002','106001','106002','106003','106004','106005'\
                                                    ,'106006','106011','106014','107001','107002','107003','107004','107005','107006','107007','108001','108002'\
                                                        ,'108003','108004','109001','109002','110001','110002','110003','110004','110005','110006','110007','111001','112001']]

# df = df.drop(columns = ['Unnamed: 0','id'])
# df = df.drop(columns = ['Company_Industry_Classification_Code','cus_Dividends_per_share_original','Dividends',
#                         'Avg_Diluted_Shares_Outstanding'])

##################
# ticker_counts = df['Ticker'].value_counts()
# ticker_list = ticker_counts[ticker_counts>=20].index.tolist()

### Fill nan
df = df.fillna(1e-6)
df = utils.replace_inf_in_df_with_0(df)
### Debug
# print(df[df['Ticker']=='NVDA'])
class Custom_Dataset_Date(Dataset):
    '''
    Return particular period from dataset
    '''
    def __init__(self,df_x=df, df_y=df_target, window=12,target_window =12,mask = None, eval=False):
        ### Initialize time window for input
        self.window = window
        self.target_window = target_window
        self.eval = eval
        ### Initialize mask
        if mask:
            self.mask = df_x[df_x['Ticker'].isin(mask)].copy()
        else:
            self.mask = df_x
            self.df_y = df_y

        self.mask = self.mask.set_index(['date', 'Ticker'])

        ### Initilaize ticker list
        ticker_counts = self.mask.reset_index()['Ticker'].value_counts()
        # self.ticker_list = ticker_counts[ticker_counts >= (self.window+self.target_window)].index.tolist() # use this when target is from simfin
        self.ticker_list = ticker_counts[ticker_counts >= (self.window)].index.tolist() # use this when target is from yf prices
        
        ### Find interaction between simfin and yf returns
        self.ticker_list = list(set(self.ticker_list).intersection(self.df_y.columns))

        ### Get date list
        self.date_list_x = self.mask.index.get_level_values('date').unique().sort_values()
        self.df_y = self.df_y.set_index(['date'])
        self.date_list_y = self.df_y.index.get_level_values('date').unique().sort_values()
        self.date_list = self.date_list_y[self.date_list_y>= min(self.date_list_x)]
        self.df_y = self.df_y[self.df_y.index.get_level_values('date').isin(self.date_list)].reset_index()

    def __getitem__(self, index):
        ### filter df_x by index date
        index_date = self.date_list[index:(index+self.window)]
        index_date_last = index_date[-1]
        item_df = self.mask[self.mask.index.get_level_values('date').isin(index_date)]

        ### Get the usable ticker list for df_x
        item_ticker_list = self.mask[self.mask.index.get_level_values('date').isin(index_date)].reset_index()['Ticker']\
            .value_counts()
        item_ticker_list = item_ticker_list[item_ticker_list==self.window].index.tolist() # Choose tickers with the value count of self.window
        
        ### Get the usable ticker list from yf
        item_ticker_list_y = self.df_y[self.df_y['date'].isin([index_date_last])].copy().reset_index(drop=True)
        if self.eval==False: # if self.eval is False, drop the dates with no target value
            item_ticker_list_y = item_ticker_list_y.replace(1e-6, np.nan).dropna(how='all',axis=1)
        item_ticker_list_y = item_ticker_list_y.columns

        ### Combine ticker list to make the final ticker list
        # if self.eval: # if self.eval is true, we only use item_ticker_list 
            # item_ticker_list = list(set(item_ticker_list))
        # else: # if self.eval is false, we use item_ticker_list and item_ticker_list_y to make sure y exists for all tickers 
        item_ticker_list = list(set(item_ticker_list).intersection(item_ticker_list_y))

        ### Define x
        dataset_x = item_df[item_df.index.get_level_values('Ticker').isin(item_ticker_list)].drop(columns=['Share_Price'])

        return dataset_x
    def __len__(self):
        if self.eval==False:
            return len(self.date_list) - (self.window + self.target_window) #+ 1
        else:
            return len(self.date_list_x) - (self.window) + 1

    def get_date(self, index):
        if self.eval==False:
            return self.date_list[(index+self.window-1)]
        else:
            return self.date_list_x[(index+self.window-1)]
class Custom_dataset_ticker(Dataset):
    '''
    Return particular period from dataset
    '''
    def __init__(self,df_x=None, ticker_list=None,df_y=df_target, window=12,target_window =12,mask = None):
        ### Initialize time window for input
        self.window = window
        self.target_window = target_window
        ### Initialize ticker list
        self.ticker_list = ticker_list

        ### Initialize dataset
        self.df_x = df_x
        self.df_y = df_y
    def __getitem__(self, index):
        # filter only one ticker
        dataset_x = self.df_x[self.df_x.index.get_level_values('Ticker') ==self.ticker_list[index]].reset_index()
        # Get y
        target_date=dataset_x['date'].iloc[-1]
        target_ticker=dataset_x['Ticker'].iloc[-1]
        # print('target_date,target_ticker',target_date,target_ticker)
        dataset_y = df_target[df_target.date==target_date][target_ticker]
        # Drop date and Ticker columns
        # print('dataset_x',dataset_x)
        dataset_x = dataset_x.drop(columns=['date','Ticker'])

        # Turn into tensor
        tensor_x = torch.tensor(np.nan_to_num(dataset_x.values.astype(dtype='float32'))).to(device)
        # Todo let me figure out if this flatten operation is needed
        tensor_x = torch.flatten(torch.t(tensor_x)).to(device)
        tensor_y = torch.tensor(dataset_y.values).to(device)

        return tensor_x,tensor_y
    def __len__(self):
        return len(self.ticker_list)
    def getitem_in_df(self,index):
                # filter only one ticker
        dataset_x = self.df_x[self.df_x.index.get_level_values('Ticker') ==self.ticker_list[index]].reset_index()
        # Get y
        target_date=dataset_x['date'].iloc[-1]
        target_ticker=dataset_x['Ticker'].iloc[-1]
        # print('target_date,target_ticker',target_date,target_ticker)
        dataset_y = df_target[df_target.date==target_date][target_ticker]
        # Drop date and Ticker columns
        # print('dataset_x',dataset_x)
        # dataset_x = dataset_x.drop(columns=['date','Ticker'])

        return dataset_x,dataset_y
class Agent():
    #Todo Try nested cross-validation
    def __init__(self,load_dir = None,save_dir = None,first_step = 0,use_alpha=True,weight_file_name = None):
        # Create an experiment with your api key
        self.experiment = Experiment(
            api_key="3dZ6osa1d98oDZoB5U0ieOfqy",
            project_name="compai",
            workspace="yudaifurukawa",
        )

        self.experiment.log_parameters(params)
        self.experiment.log_parameters(val_params)

        ### Initialize Time
        start_time = time.time()

        self.first_step=first_step
        self.load_dir=load_dir
        self.weight_file_name = weight_file_name
        self.y_window = 12
        self.x_window = 12

        if save_dir == None:
            sys.exit('save_dir is missing')

        # Initiate directory to save weights and logs
        self.save_dir_log = './logs/Analyze/'+save_dir
        self.save_dir_weight = './saved_model/'+save_dir

        ###############################
        ### Initialize Dataset
        ###############################
        # Initialize self.dataset_daterange dataset
        self.dataset_daterange = Custom_Dataset_Date(df_x=df,df_y=df_target)

        # Initialize self.dataset_ticker dataset
        ticker_list=self.dataset_daterange[0].reset_index()['Ticker'].unique().tolist()
        self.dataset_ticker=Custom_dataset_ticker(self.dataset_daterange[0],ticker_list=ticker_list)

        ###############################
        ### Initilize model
        ###############################
        self.model = net(len(self.dataset_ticker[0][0])).to(device)

        ### Initilize self.optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)

        ### Print Time
        print(f'Initialization execution time = {(time.time() - start_time)/60} min')

    def run(self):
        start_time = time.time()
        ###############################
        ### Innitilize Logger
        ###############################
        # writer = tensorboard.SummaryWriter(log_dir=self.save_dir_log)

        ### Load weight
        try:
            self.model.load_state_dict(torch.load('./saved_model/' + self.load_dir + '/' + self.weight_file_name))
            print('Loaded ./saved_model/' + self.load_dir + '/' + self.weight_file_name)
        except:
            print('Did not load weights...')
            pass

        ### Loop
        # index_date = len(self.dataset_daterange)-((self.x_window-1)*3+self.y_window)-1
        index_date = len(self.dataset_daterange) - self.x_window #- 1
        print(self.dataset_daterange.get_date(index_date))
        self.experiment.log_parameter("train_date",self.dataset_daterange.get_date(index_date))
        self.experiment.log_parameter("val_date",self.dataset_daterange.get_date(index_date+self.x_window))
        print('len(self.dataset_daterange)',len(self.dataset_daterange))

        # print("len(self.dataset_daterange)-((self.x_window-1)*3+self.y_window):",len(self.dataset_daterange)-((self.x_window-1)*3+self.y_window))
        # Todo let the algorism run from the latest data
        experiment_step=0
        with self.experiment.train(): # Train for comet
            while index_date > 10:
                # reset_parameters
                self.model.reset_parameters()
                
                ### Initiate memory
                # it seems it's taking a lot of time when maxlen = 1000
                outputs = deque(maxlen=500)
                actuals = deque(maxlen=500)
                outputs_val = deque(maxlen=500)
                actuals_val = deque(maxlen=500)

                ### Set Initilial value to best_loss_val
                best_loss_val = float('inf')
                best_loss = float('inf')
                best_epoch = 0
                best_ii = 0
                best_r2_val = float('-inf')

                ########################
                ### Validation set
                ########################
                validation_date_index = index_date + self.x_window 
                # validation_date_index = index_date + (self.x_window - 1) + self.y_window
                validation_ticker_list = self.dataset_daterange[validation_date_index].reset_index()[
                    'Ticker'].unique().tolist()
                print('self.dataset_daterange.get_date(validation_date_index)',self.dataset_daterange.get_date(validation_date_index))
                validation_set = Custom_dataset_ticker(self.dataset_daterange[validation_date_index],
                                                    ticker_list=validation_ticker_list)
                print('validation_set[len(validation_set)-1]',validation_set.getitem_in_df(len(validation_set)-1))
                validation_generator = data.DataLoader(validation_set, **params)

                ####################################################
                ### Training
                ####################################################
                for ii in range(1000): # The number of times to iterate random_range
                    
                    # random_range = list(range(index_date+1,index_date-6,-1)) # Use only 7 dates
                    random_range = list(range(index_date + 1, 1, -1))
                    random.shuffle(random_range)
                    print(f'max_date: {max(random_range)}, {self.dataset_daterange.get_date(max(random_range))}, \
                        random_range: {random_range}',random_range, )

                    num_date_i=0
                    
                    for i in random_range:
                        
                        num_date_i+=1
                        print(f'index_date: {i}... , date: {self.dataset_daterange.get_date(i)}... loop: {ii}/5000, date_running: {i}, \
                            completed... {num_date_i/index_date*100} %, training time = {(time.time() - start_time)/60} min,\
                            best_r2_val {best_r2_val}, best_loss_val {best_loss_val}')
                        ##################
                        # Train set
                        ##################
                        train_date_index= i
                        train_ticker_list = self.dataset_daterange[train_date_index].reset_index()['Ticker'].unique().tolist()
                        train_set = Custom_dataset_ticker(self.dataset_daterange[train_date_index], ticker_list=train_ticker_list)

                        #Todo print data caracteristics for train and validation set

                        if len(train_ticker_list)<8:
                            drop_last=False
                        else:
                            drop_last=True
                        train_generator = data.DataLoader(train_set, drop_last=drop_last, **params)
                        print(f'train generator ready in {(time.time() - start_time)/60} min')
                        for epoch in range(1):
                            ### Train
                            
                            self.model.train()
                            for local_batch, local_labels in train_generator: # Todo this part seems to be taking time
                                local_batch, local_labels = local_batch.to(device), local_labels.to(device)
                                self.optimizer.zero_grad()
                                output = self.model(local_batch).flatten().unsqueeze(1).float()
                                loss = F.mse_loss(output.float(), local_labels.float())
                                # Todo, maybe I shouldn't use mse_loss?
                                
                                loss.backward()
                                self.optimizer.step()

                                ### Store result
                                
                                outputs.extend(output.flatten().detach().numpy())
                                actuals.extend(local_labels.flatten().detach().numpy())

                            print(f'training finished in {(time.time() - start_time)/60} min')
                            ##############################################
                            ### Evaluate
                            ##############################################
                            self.model.eval()

                            ## Get one batch for evaluation > val is calculated only based on 0ne batch
                            # iterator_val = iter(validation_generator)
                            # local_batch, local_labels = iterator_val.next()

                            for local_batch, local_labels in validation_generator: # Todo this part seems to be taking time too
                                local_batch, local_labels = local_batch.to(device), local_labels.to(device)
                                output_val = self.model(local_batch).flatten().unsqueeze(1).float()

                                # Validation loss
                                loss_val = F.mse_loss(output_val.float(), local_labels.float())

                                # extend deque
                                outputs_val.extend(output_val.flatten().detach().numpy())
                                actuals_val.extend(local_labels.detach().numpy())
                            print(f'validation finished in {(time.time() - start_time)/60} min')

                            ### Calculate R2  #####################
                            r2 = r2_score(np.array(actuals), np.array(outputs))
                            r2_val = r2_score(np.array(actuals_val), np.array(outputs_val))

                            output_array = [np.array(outputs),np.array(actuals)]
                            np.savetxt("deque.csv", output_array, delimiter=",")


                            # Calculate Average
                            average_actuals = np.mean(actuals)
                            average_outputs = np.mean(outputs)
                            average_actuals_val = np.mean(actuals_val)
                            average_outputs_val = np.mean(outputs_val)

                            # calculate std
                            std_actuals = np.std(actuals)
                            std_outputs = np.std(outputs)
                            std_actuals_val = np.std(actuals_val)
                            std_outputs_val = np.std(outputs_val)

                            print(f'r2 calculated in {(time.time() - start_time)/60} min')

                            #######################################
                            ### Logging to comet.ai
                            #######################################
                            experiment_step+=1
                            print('experiment_step',experiment_step)
                            self.experiment.log_metric("r2", r2, step=experiment_step)
                            self.experiment.log_metric("r2_val", r2_val, step=experiment_step)
                            self.experiment.log_metric("loss", loss, step=experiment_step)
                            self.experiment.log_metric("loss_val", loss_val, step=experiment_step)
                            
                            # Average
                            self.experiment.log_metric("average_actuals", average_actuals, step=experiment_step)
                            self.experiment.log_metric("average_outputs", average_outputs, step=experiment_step)
                            self.experiment.log_metric("average_actuals_val", average_actuals_val, step=experiment_step)
                            self.experiment.log_metric("average_outputs_val", average_outputs_val, step=experiment_step)
                            
                            # std
                            self.experiment.log_metric("std_actuals", std_actuals, step=experiment_step)
                            self.experiment.log_metric("std_outputs", std_outputs, step=experiment_step)
                            self.experiment.log_metric("std_actuals_val", std_actuals_val, step=experiment_step)
                            self.experiment.log_metric("std_outputs_val", std_outputs_val, step=experiment_step)

                            # Histogram
                            self.experiment.log_histogram_3d(outputs,"outputs",experiment_step)
                            self.experiment.log_histogram_3d(actuals,"actuals",experiment_step)
                            self.experiment.log_histogram_3d(outputs_val,"outputs_val",experiment_step)
                            self.experiment.log_histogram_3d(actuals_val,"actuals_val",experiment_step)
                            
                            # # Log data
                            # self.experiment.log_text(outputs,step=experiment_step)
                            # self.experiment.log_text(actuals,step=experiment_step)

                            #######################################


                            ## Save the best loss
                            if epoch % 1 == 0:
                                ### R2 was better than mse because some data point doesn't have much variance
                                # if r2_val > best_r2_val: # Maybe I can use mse using the list for r2 and r2_score.
                                if loss_val < best_loss_val:
                                    print('r2_val is greater than  best_r2_val')
                                    print('epoch...{}, r2_val...{}, loss_val...{}, average...{}, average_val... {}'
                                        .format(epoch, r2_val, loss_val.item(),average_actuals_val, average_outputs_val))
                                    # Update the best loss and epoch
                                    best_loss_val = loss_val
                                    best_r2_val = r2_val
                                    best_epoch = epoch
                                    best_ii = ii
                                    # if epoch > -1: #Not saving till reaching 100 epochs
                                    #     if r2_val > 0.3:
                                    print('saving weight for the best loss_val...')
                                    torch.save(self.model.state_dict(), self.save_dir_weight+'/model_weights_best_loss_val.pth')
                                if loss < best_loss:
                                    best_loss = loss
                                    print('saving weight for the best loss...')
                                    torch.save(self.model.state_dict(), self.save_dir_weight+'/model_weights_best_loss.pth')

                            ### Print results and save in log
                            if ii % 100 == 0:
                                ### Print time
                                print(f'index_date {index_date}, i {i}, epoch {epoch}, training time = {(time.time() - start_time)/60} min')

                                ### Logger to Tensorboard
                                # step = i*1e3 + epoch + self.first_step  # default value for first_step is 0
                                # info = {'loss': loss.item(), 'loss_val': loss_val.item(), 'r2_score': r2, 'r2_score_val': r2_val}
                                # for tag, value in info.items():
                                #     # logger.scalar_summary(tag, value, step)
                                #     writer.add_scalar(tag, value, step)

                                # for tag, value in self.model.named_parameters():
                                #     tag = tag.replace('.', '/')
                                #     writer.add_histogram(tag, value.data.cpu().numpy(), step)
                                #     writer.add_histogram(tag + '/grad', value.grad.data.cpu().numpy(), step)

                                # print the best epoch
                                print(f'current r2...{r2}, loss...{loss}')
                                print(f'current r2_val...{r2_val}, loss_val...{loss_val}')
                                print(
                                    f'best index_date...{index_date}:best_epoch ...{best_epoch}:best_loop...{best_ii}, best_r2_val...{best_r2_val}, best_loss_val...{best_loss_val}')
                                # Print result
                                print("epoch {}, loss {:.6f}, loss_val {:.6f}".format(epoch,loss,loss_val))
                                print("r2 {:.6f}, r2_val {:.6f}".format(r2,r2_val))
                                print('train_data (outputs, actuals): ',
                                    list(zip(itertools.islice(outputs,0,10), itertools.islice(actuals,0,10))))
                                print('val_data (outputs, actuals): ',
                                    list(zip(itertools.islice(outputs_val,0,10), itertools.islice(actuals_val,0,10))))

                # Next index_date
                # index_date-=10
                index_date -= 9999
                # Print best epoch, best r2 and best loss
                print('best epoch ...{} current best_r2_val...{}, best_loss_val... {}'
                    .format(best_epoch, best_r2_val, best_loss_val))

        # writer.close()
    def combine_model_weights(self):
        num_files = len(os.listdir(self.save_dir_weight))
        weight_final = dict()

        # Su mall the weights, bias etc
        for file in os.listdir(self.save_dir_weight):
            # Check if weight_final is empty
            if not weight_final:
                weight_final=torch.load(self.save_dir_weight + '/' + file)
                print('if not weight_final:')
            else:
                weight_temp=torch.load(self.save_dir_weight + '/' + file)
                for key in weight_final:
                    print('weight_final[key]',weight_final[key])
                    print('weight_temp[key]',weight_temp[key])
                    weight_final[key]+=weight_temp[key]
                    print('weight_final[key]+=weight_temp[key]',weight_final[key])

        # Devide by the number of the files
        for key in weight_final:
            print('weight_final[key]', weight_final[key])
            weight_final[key]=np.true_divide(weight_final[key],num_files)
            print('weight_final[key/5]', weight_final[key])

        # save
        torch.save(weight_final, self.save_dir_weight + '/model_weights_final.pth')
    def test(self):
        ### Initialize lists
        outputs_test = list()
        actuals_test = list()

        ### Initialize Test set
        ### test set
        test_index = len(self.dataset_daterange)
        test_ticker_list = self.dataset_daterange[test_index].reset_index()['Ticker'].unique().tolist()
        test_set = Custom_dataset_ticker(self.dataset_daterange[test_index], ticker_list=test_ticker_list)
        test_generator = data.DataLoader(test_set, **params)

        ### Load model weights
        self.model.load_state_dict(torch.load(self.save_dir_weight + '/model_weights_final.pth'))
        self.model.eval()
        # https://stackoverflow.com/questions/57386851/how-to-get-entire-dataset-from-dataloader-in-pytorch
        # for i in range(len(test_set)):
        for local_batch, local_labels in test_generator:
            local_batch, local_labels = local_batch.to(device), local_labels.to(device)
            output_test = self.model(local_batch).flatten().unsqueeze(1).float()

            # Validation loss
            loss_test = F.mse_loss(output_test.float(), local_labels.float())


            # extend deque
            outputs_test.extend(output_test.flatten().detach().numpy())
            actuals_test.extend(local_labels.detach().numpy())

            ### Calculate R2 Score
            r2_test = r2_score(actuals_test, outputs_test)
            print('r2_test',r2_test)
class Evaluate_Agent():
    def __init__(self, save_dir = None, load_dir = None, save = True,use_alpha = False):
        ### Initilize save dir
        self.save_dir = './saved_model/' + save_dir
        self.load_dir = './saved_model/' + load_dir 
        self.save = save
        self.use_alpha = use_alpha
        self.eval_df = pd.DataFrame()

        ###############################
        ### Initialize Dataset
        ###############################
        # Initialize self.dataset_daterange dataset
        self.dataset_daterange = Custom_Dataset_Date(df_x=df,df_y=df_target,eval=True)
        # print('self.dataset_daterange.get_date(len(self.dataset_daterange)',self.dataset_daterange.get_date(len(self.dataset_daterange)))
    

        # Initialize self.dataset_ticker dataset
        ticker_list=self.dataset_daterange[0].reset_index()['Ticker'].unique().tolist()
        self.dataset_ticker=Custom_dataset_ticker(self.dataset_daterange[0],ticker_list=ticker_list)
    def run(self, weight_file):
        # Initialize model
        self.model = net(len(self.dataset_ticker[0][0])).to(device)
        ### Load weight
        print("loading weight from ...", self.load_dir +'/' + weight_file)
        self.model.load_state_dict(torch.load(self.load_dir + '/' + weight_file))
        # ### Load weight
        # print("loading weight from ...", self.load_dir)
        # self.model.load_state_dict(torch.load(self.load_dir))

        ## Evaluate
        self.model.eval()
        print('len(self.dataset_daterange)',len(self.dataset_daterange),self.dataset_daterange.get_date(len(self.dataset_daterange)-1))

        for date_index in range(len(self.dataset_daterange)-1,0,-1):
            self._eval_one_date(date_index)
            self.eval_df = self.eval_df.reset_index(drop=True)

            if self.save:
                self.eval_df.to_csv(self.save_dir + '/' + weight_file +'_eval.csv')

            # Todo add date to csv
            # Todo remove blanket -> done?

        print(self.eval_df)
    def _eval_one_date(self,date_index):
        print('running date_index...',date_index,self.dataset_daterange.get_date(date_index))

        # Initiate dataset
        eval_ticker_list = self.dataset_daterange[date_index].reset_index()['Ticker'].unique().tolist()
        eval_set = Custom_dataset_ticker(self.dataset_daterange[date_index], ticker_list=eval_ticker_list)
        print('eval_ticker_list',eval_ticker_list)

        
        y_list = []
        y_pred_list=[]

        # evaluation loop
        print("len(eval_set)...",len(eval_set))
        # eval_generator = data.DataLoader(eval_set, batch_size=len(eval_set),num_workers=4,shuffle=False)
        eval_generator = data.DataLoader(eval_set, batch_size=3,num_workers=4,shuffle=False)
        for local_batch, local_labels in eval_generator:
            local_batch, local_labels = local_batch.to(device), local_labels.to(device)
            output_eval = self.model(local_batch).flatten().unsqueeze(1).float()
            # https://discuss.pytorch.org/t/get-value-out-of-torch-cuda-float-tensor/2539/5
            # https://stackoverflow.com/questions/57727372/how-do-i-get-the-value-of-a-tensor-in-pytorch
            y_list.extend(local_labels.detach().numpy())
            y_pred_list.extend(output_eval.detach().numpy())

        # Create Dataframe Ticker|Date|y|y|pred|r2
        eval_df_tmp = pd.DataFrame(list(zip(eval_ticker_list,y_list,y_pred_list)),
                                   columns=['Ticker','y','y_pred'])
        # Add date column
        eval_df_tmp['date']= self.dataset_daterange.get_date(date_index)

        # Get Share_Price
        eval_df_tmp = pd.merge(eval_df_tmp,df[['Ticker','date','Share_Price']], on=['Ticker','date'],how='left')

        # Replace 0.000001 with -99
        eval_df_tmp['y']=eval_df_tmp['y'].replace(0.000001,-99)

        # Reorder Columns
        eval_df_tmp=eval_df_tmp[['Ticker','date','Share_Price','y','y_pred']]
        print('self.eval_df with Share_Price',eval_df_tmp)



        # Concat
        self.eval_df = pd.concat([self.eval_df,eval_df_tmp])


        ### Change datatypes
        self.eval_df['y']=self.eval_df['y'].astype('float')
        self.eval_df['y_pred'] = self.eval_df['y_pred'].astype('float')
        # self.eval_df['date'] = self.eval_df['date'].astype('float')
        self.eval_df['date']=pd.to_datetime(self.eval_df['date'])
        print(self.eval_df.dtypes)
        print('self.eval_df',self.eval_df)

        return self.eval_df

if __name__ == '__main__':
    print('Starting main...')
    dir = '220416_gc' #20210807_rank
    # a = Agent(save_dir = dir, load_dir = dir,first_step=1,use_alpha=True,weight_file_name='model_weights_best_loss.pth')
    # a.run()
    
    # Initialize self.dataset_daterange dataset
    # dataset_daterange = Custom_Dataset_Date(df_x=df,df_y=df_target)

    # Initialize self.dataset_ticker dataset
    # ticker_list=dataset_daterange[0].reset_index()['Ticker'].unique().tolist()
    # dataset_ticker=Custom_dataset_ticker(dataset_daterange[0],ticker_list=ticker_list)
    # print(len(dataset_ticker),dataset_ticker[len(dataset_ticker)-1])
    
    # b = Custom_Dataset_Date()
    # print('len(b)',len(b),b.get_date(len(b)))
    # b = Custom_Dataset_Date(eval=True)
    # print('eval len(b)',len(b),b.get_date(len(b)))
    # print(b[30])
    # print('a.dataset_daterange',a.dataset_daterange[41],a.dataset_daterange[87])

    for weight_file in ['model_weights_best_loss.pth','model_weights_best_loss_val.pth']: #'model_weights_20.pth','model_weights_30.pth','model_weights_40.pth','model_weights_final.pth']:
        b = Evaluate_Agent(save_dir=dir,load_dir=dir,save=True,use_alpha=True)
        b.run(weight_file=weight_file)
    # b._eval_one_date(date_index=94)
    # b = Custom_Dataset(mask=['AAPL'],use_alpha=False)
    # c = Custom_Dataset_Eval(ticker='AAPL',use_alpha=False)
    # print(df[df.date<'20000101'].head())
    #Todo create new dataset

# conn.commit()
# conn.close()