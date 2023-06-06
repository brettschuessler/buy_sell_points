#core
import numpy as np
import pandas as pd
import seaborn as sns
import json
import matplotlib as mpl
import matplotlib.pyplot as plt
from pandas.tseries.offsets import DateOffset
mpl.rcParams['figure.figsize'] = (25,20)
#from tqdm import tqdm
import tqdm.notebook as tq

from sklearn.neighbors import KNeighborsClassifier
from sklearn.kernel_ridge import KernelRidge
from sklearn import svm
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

#Helpers
from sklearn import preprocessing as pp
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.utils import shuffle as dual_shuffle
from sklearn.neighbors import KernelDensity
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
import scipy
from scipy import stats
import random
import datetime

# The DF_JACKET class allows me to pull OHLC data from yfinance for a particualr equity over a particualr time period and:
# 1) find the optimal buy sell points, given the number of transactions for a day
# 2) pull the previous k days of pct change 'data' before each point, ('data' is any creative combo of OHLC)
# 3) generate a nice dataframe with this past data as a vector aligned with each day and the b/s value


class DF_JACKET:
    def __init__(self, df_in, n_day=3, data_dim=4, source_type='oc_avg', its=1e4):
        #__ Takes a dataframe with a datetime index __
        self.df = df_in
        self.df['BS'] = 'Hold'

        self.n = n_day
        self.its = int(its)
        self.size = self.df.shape[0]
        self.bounds = {'s_month':self.df.index[0].month, 's_day':self.df.index[0].day,
                       'e_month':self.df.index[self.size-1].month, 'e_day':self.df.index[self.size-1].day}
        self.vtd = self.find_trading_days()


        self.fbi = [] # first buy index, per day

        self.k = data_dim
        self.df['S'] = self.maps_OHLC_to_scalar(map_type=source_type)
        #self.df['P'] = self.maps_OHLC_to_scalar(map_type=source_type, raw=True)
        self.df['P'] = self.df.Open

        self.t_data = None

    #______________________________________________________________________________________________________________
    # ____________________________________ Code for a single day __________________________________________________

    def daily_df(self, _df,  month=None, day=None, tf=False):
        # returns the daily dataframe for month, day
        # tf keyword allows this function to act as a logic unit to find valid trading days
        tdf = _df[(_df.index.day == day) & (_df.index.month == month)]
        if tf:
            if not tdf.empty:
                return True
            else:
                return False
        else:
            if not tdf.empty:
                return tdf
            else:
                raise ValueError

    def find_minmax(self, day_df, column='Open'):
        ary = day_df[column].values
        mm_vec = [] #time ordered local min max vector
        max_idx_vec = [] # bool mask for local maxima over mm_vec
        min_idx_vec = None # bool mask for local minima over mm_vec
        idx_vec = [] # index of each max/min over the whole day
        for j in range(0, ary.shape[0]):
            #__ Edge Cases __
            if j==0:
                idx_vec.append(j)
                mm_vec.append(ary[0])
                if ary[1] < ary[0]:
                    max_idx_vec.append(True)
                else:
                    max_idx_vec.append(False)
            #print(ary.shape[0]-1)
            elif j==(ary.shape[0]-1):
                idx_vec.append(j)
                mm_vec.append(ary[j])
                if ary[j-1] < ary[j]:
                    max_idx_vec.append(True)
                else:
                    max_idx_vec.append(False)
            #___________________________________________
            #__ The Middle Cases ___
            elif (ary[j-1]<ary[j]) & (ary[j+1]<ary[j]): # Finds local maxima
                idx_vec.append(j)
                mm_vec.append(ary[j])
                max_idx_vec.append(True)
            elif (ary[j-1]>ary[j]) & (ary[j+1]>ary[j]): # Find local minima
                idx_vec.append(j)
                mm_vec.append(ary[j])
                max_idx_vec.append(False)

        max_idx_vec = np.array(max_idx_vec)
        min_idx_vec = ~max_idx_vec
        return {'max_mask':max_idx_vec.reshape(1,-1), 'min_mask':min_idx_vec.reshape(1,-1),
               'max_min':np.array(mm_vec),'idx':idx_vec}

    def gen_pwd_msk(self, **kwargs):
        #generates the pairwise diff matrix of all max/mins
        #generates the mask of possible buy/sell pairs for said pwd matrix
        for k,v in kwargs.items():
            if k == 'min_mask':
                min_msk = v
            if k == 'max_min':
                mx_mn = v
            if k == 'idx':
                df_idx = v

        mm_sz = mx_mn.shape[0] #number of maxes/mins in a day
        A = np.tile(mx_mn, (mm_sz,1))
        M = A - A.T #pairwise difference matrix of all maxes/mins
        msk = np.triu(min_msk.T*~min_msk) #mask of possible max/min pairs that make sense temporally

        ridx = np.indices((mm_sz,mm_sz)) #quick way of generating mtx indices
        idxl = list(zip(ridx[0][msk], ridx[1][msk])) #list of matrix index tuples allowed under mask
        return M, msk, idxl, df_idx

    #____________ Runs multiple times per run for a day________________________________
    def gen_n_indices(self, idx_list):
        ol = []
        idl = idx_list
        for k in range(0, self.n):
            try:
                idx = self.rand_pick_idxlist(idl)
                ol.append(idx)
                idl = self.tuple_filter(idl, *idx)
            except:
                #print('Not enough choices left to satisfy n= %s' %(str(self.n)))
                pass
        return ol

    def rand_pick_idxlist(self, ij_list):
        # picks a random element of allowed choices from allowed list
        # potential integration of a weighted choice for effiecieny to be explored
        #print(len(ij_list))
        return random.choice(ij_list)

    def tuple_filter(self, ij_list, i, j):
        # takes the list of tuples of indices from the pwd matrix of allowed choices
        # outputs the new truncated list of allowed choices based on previous choice
        ol = []
        for idx in ij_list:
            if (idx[0]>i) & (idx[1]>j) & (idx[0]>j) & (idx[1]>i):
                ol.append(idx)
        return ol
    #___________Runs each run for a day _____________________
    def calc_daily_profit(self, M, pw_indices):
        ol = []
        for idx in pw_indices:
            ol.append(M[idx[0], idx[1]])
        #print('Number of BS points: %s' %(str(len(ol))))
        return sum(ol), pw_indices

    #___________Runs once per day _____________________
    def mm_tuples_to_df_index(self, tuples_idx, df_idx):
        # Takes the tuple indices from the pwd matrix of mins/maxes and gets the corresponding
        # indices from the overall dataframe
        tmp_buy_idx = [e[0] for e in tuples_idx] # index for buy actions under the pwd matrix
        tmp_sell_idx = [e[1] for e in tuples_idx]
        buy_idx = np.array(df_idx)[tmp_buy_idx] # index for buy actions under overall daily df
        sell_idx = np.array(df_idx)[tmp_sell_idx]
        return buy_idx, sell_idx


    def run_day(self, month=10, day=8, its= 10000, p_bar=False):
        df_ = self.daily_df(self.df, month, day)
        M, msk, idl, df_idx = self.gen_pwd_msk(**self.find_minmax(df_))
        profit_list = []
        indices_list = []
        #for j in tq.tqdm(range(0,its)): # progress bar
        for j in self.tq_wrap(range(0,its), display=p_bar): # progress bar
            profit, pw_indices = self.calc_daily_profit(M, self.gen_n_indices(idl))
            profit_list.append(profit)
            indices_list.append(pw_indices)
        pl_argmax = np.argmax(np.array(profit_list))
        buy_idx, sell_idx = self.mm_tuples_to_df_index(indices_list[pl_argmax], df_idx)


        self.fbi.append(buy_idx[0])

        b_tidx = list(df_.iloc[buy_idx].index)
        self.df.loc[b_tidx, 'BS']='Buy'
        s_tidx = list(df_.iloc[sell_idx].index)
        self.df.loc[s_tidx, 'BS']='Sell'

        return None #self.df.loc[tidx]
        #indices_list[pl_argmax], max(profit_list), sum(profit_list)/len(profit_list), df_idx
        #return M#df.iloc[buy_idx]

    def tq_wrap(self, iterable, display=False):
        # allows for passing a boolean to display progress bars
        if display:
            return tq.tqdm(iterable)
        elif not display:
            return iterable


    #_______________________________________________________________________________________________________________
    #____________________________________ Code to run over multiple days ___________________________________________

    def run_days(self, p_bar=True):
        for date in self.tq_wrap(self.vtd, display=p_bar):
            #self.run_day(month=date[0],day=date[1])
            self.run_day(*date, its=self.its)

    def find_valid_months(self):
        # generates a list of valid months for the dataset
        #---- this will cause problems if I run with more than a years data -----
        if self.bounds['s_month'] < self.bounds['e_month']:
            return list(range(self.bounds['s_month'], self.bounds['e_month']+1))
        else:
            return list(range(self.bounds['s_month'],13)) + list(range(1, self.bounds['s_month']+1))
            # This looks childish concatenating lists with '+' but as per the nutjobs with runtime graphs on stack exchange ; this is absolutely optimal python

    def find_trading_days(self):
        # finds all trading days in the dataset
        vtd = []
        for m in self.find_valid_months():
            for d in range(1,31):
                if self.daily_df(self.df, month=m, day=d, tf=True):
                    vtd.append([m, d])
        return vtd

    #______________________________________________________________________________________________________________________
    #________________________________________ Constructs data for algorithm _______________________________________________

    def maps_OHLC_to_scalar(self, map_type='oc_avg', raw=False):
        if map_type=='oc_avg':
            series_out = (self.df.Open + self.df.Close)/2
        if raw:
            return series_out
        else:
            return series_out.pct_change()


    def populate(self, p_bar=False):
        ol = []
        for date in self.tq_wrap(self.vtd, display=p_bar):
            ol.append(self._populate(date))
        self.t_data = pd.concat(ol)
        #return pd.concat(ol)


    def _populate(self, date):
        daily_df = self.daily_df(self.df, *date)
        nddf = pd.DataFrame(np.concatenate(self.slice_n_expand(daily_df)[:-(self.k+1)], axis=0), index=daily_df.index[self.k+1:].copy())
        nddf['BS'] = self.df.BS
        nddf['P'] = self.df.P
        return nddf

    def slice_n_expand(self, _df):
        # takes daily dataframe and expands the 'S' column into (1xk), the k lagging datapoints for a particular day
        arr = _df.S.values
        big_arr = []
        for j in range(0, arr.shape[0]):
            big_arr.append(arr[j+1:j+1+self.k].reshape(1,-1))
        return big_arr
