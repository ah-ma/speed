from numba import njit
import pandas as pd
from pandas.tseries.offsets import BDay
import numpy as np
import time as t
import timeit


@njit(nogil=True)
def calc_func_vol(a, b, var):
    n = len(var)
    res = np.empty(n)
    res[0] = var[0]
    fact = 1. - (a / b)
    fact_1 = 0.
    for i in range(1, n):
        fact_1 = fact * res[i - 1] + var[i]
        res[i] = fact_1
    return res


@njit(nogil=True)
def calc_func_decay(a, b, n):
    res = np.empty(n)
    fact_1 = b
    fact_2 = 0.
    res[0] = fact_2
    fact_2 = a * res[0] + (1. - a) * 1.
    res[1] = fact_2
    fact_2 = a * res[1] + (1. - a) * fact_1
    res[2] = fact_2
    for i in range(3, n):
        fact_1 *= b
        fact_2 = a * res[i - 1] + (1. - a) * fact_1
        res[i] = fact_2
    return res


def prep_rv(in_df, asset_col, params_decay_default):
    in_df["pctChange"] = in_df[asset_col].pct_change().add(1)
    in_df["log_pctChange"] = np.log(in_df["pctChange"])
    in_df["intraday_var"] = np.power(in_df["log_pctChange"], 2)

    max_daily_var = np.percentile(in_df["intraday_var"].fillna(0), params_decay_default["blackswan_pct"])
    
    #in_df.loc[in_df["intraday_var"] > max_daily_var, "intraday_var"] = max_daily_var
    vals = in_df['intraday_var'].values
    in_df['intraday_var'] = np.where(vals > max_daily_var, max_daily_var, vals)

    return in_df


def calcs_vol(in_df, params):
    df = in_df.copy()
    coeff = params["coeff"]
    tenor_days = params["busdays"]

    df["stdRV"] = 100 * np.sqrt(coeff * df["intraday_var"].fillna(0).rolling(window=tenor_days, center=False).sum())

    #this_initial = df.loc[0:tenor_days, "intraday_var"].sum()
    this_initial = df["intraday_var"].values[0:tenor_days+1].sum()
    
    #temp_df = df.loc[tenor_days + 1:, ["intraday_var"]]
    temp_df = pd.DataFrame({'intraday_var':df['intraday_var'].values[tenor_days + 1:]})
    
    a = calc_func_vol(params["weight_rv"], tenor_days, np.array(temp_df["intraday_var"], dtype=np.float))
    temp_df["calcs"] = this_initial * ((1 - (params["weight_rv"] / tenor_days)) **
                                       (1 + temp_df.index - temp_df.index[0])) + params["weight_rv"] * a
    
    df.loc[tenor_days, "calcs"] = this_initial
    #col_idx = df.columns.tolist().index('calcs')
    #df.iat[tenor_days, col_idx] = this_initial
    
    #df.loc[tenor_days + 1:, "calcs"] = temp_df["calcs"]    
    vals = df['calcs'].values
    vals[tenor_days + 1:] = temp_df['calcs'].values
    df['calcs'] = vals        

    df["clclRV"] = 100 * np.sqrt(coeff * df["calcs"].fillna(0))

    df["blend"] = params["weight_curr"] * df["clclRV"]
    return df


def setup_decays(in_df, params, start_date, col_atm):
    crnt_vb_decay         = params["vb_decay"]
    crnt_VTrend_decay     = params["vt_decay"]
    crnt_prctile_VH       = params["vhh_pctl"]
    crnt_High_multiple    = params["vh_multiple"]
    crnt_blended_iv       = params["iv_pct"]
    crnt_blended_rv       = params["rv_pct"]
    crnt_HighVol_perctile = params["high_vol_pctl"]
    crnt_VT_to_LY         = params["vt_to_ly"]
    crnt_low_perctile     = params["vlow_pctl"]
    
    if params["spot_start"] == "":
        crnt_VLow = np.percentile(in_df[col_atm], crnt_low_perctile)
    else:
        crnt_VLow = np.percentile(in_df.loc[in_df["date"] >= params["spot_start"], col_atm], crnt_low_perctile)

    atmIV = in_df[in_df["date"] == start_date][col_atm].iloc[0]
    candtRV_1 = in_df[in_df["date"] == start_date]["blend"].iloc[0]
    candtRV_2 = atmIV * crnt_HighVol_perctile / 100
    RV = max(candtRV_1, candtRV_2)
    VT_start = crnt_blended_iv / 100 * atmIV + crnt_blended_rv / 100 * RV

    candtVBstart_1 = crnt_VLow
    candtVBstart_2_1 = (VT_start - crnt_VLow) * crnt_VT_to_LY / 100 + crnt_VLow
    candtVBstart_2_2 = VT_start * crnt_HighVol_perctile / 100
    candtVBstart_2 = min(candtVBstart_2_1, candtVBstart_2_2)
    VB_start = candtVBstart_2

    VH_start = VT_start + (VT_start - VB_start) * crnt_High_multiple
    VB2VT_ratio = VB_start / VT_start

    this_decay = in_df.loc[in_df.date >= start_date, ["date", col_atm]].reset_index(drop=True)
    if params["iv_start"] == "":
        this_decay["value_VH_prctile"] = np.percentile(in_df[col_atm], crnt_prctile_VH)
    else:
        tmp_highpct = np.percentile(in_df.loc[in_df["date"] >= params["iv_start"], col_atm], crnt_prctile_VH)
        this_decay["value_VH_prctile"] = tmp_highpct

    this_decay.loc[0, "VT"] = VT_start
    this_decay.loc[0, "VB"] = VB_start

    a = calc_func_decay(crnt_VTrend_decay, crnt_vb_decay, len(this_decay.index))
    this_decay["VB"] = VB_start * (crnt_vb_decay ** this_decay.index)
    this_decay["VT"] = VT_start * (crnt_VTrend_decay ** this_decay.index) + VB_start * a

    this_decay["weighted_diff_VH"] = this_decay["VT"] + (this_decay["VT"] - this_decay["VB"]) * crnt_High_multiple
    this_decay["VH"] = this_decay[["weighted_diff_VH", "value_VH_prctile"]].max(axis=1)
    this_decay["VHH"] = this_decay["weighted_diff_VH"]
    this_decay.loc[this_decay["VH"] != this_decay["value_VH_prctile"], "VHH"] = np.nan
    this_decay = this_decay[["date", col_atm, "VT", "VB", "VH", "VHH"]]
    return this_decay
"""
dl = df.copy()
iy = initial_decay.copy()
del dl['vol']
if 1:
    start = time()
    for i in range(10000):
        #z= pd.DataFrame({c[0].columns[0]:np.append(c[0].values,c[1].values)})
        z=pd.concat([c[0],c[1]], axis=0, sort=False).reset_index(drop=True)
    print(time()-start)
    time.strftime("%Y-%m-%d %H:%M:%S", start_datestr+' 00:00:00' )
    datetime.strptime(start_datestr+' 00:00:00', "%Y-%m-%d %H:%M:%S")
#"""
def decay_sets_generator(df_spot, df_vol, params):
    asset_col      = [k for k in df_spot.columns if k != "date"][0]
    tenor_col      = [k for k in df_vol.columns if k != "date"][0]
    start_datestr  = min(df_spot['date'].values[0], df_vol["date"].values[0])      #min(df_spot["date"].iloc[0], df_vol["date"].iloc[0])
    start_date     = pd.to_datetime(start_datestr, format="%Y-%m-%d")
    end_datestr    = max(df_spot['date'].values[-1], df_vol["date"].values[-1])    #max(df_spot.loc[len(df_spot) - 1, "date"], df_vol.loc[len(df_vol) - 1, "date"])
    end_date       = pd.to_datetime(end_datestr, format="%Y-%m-%d")
    all_dates      = pd.DataFrame(pd.date_range(start_date, end_date, freq=BDay()).strftime("%Y-%m-%d"), columns=["date"])

    """
    def align(df_dates, df_spot, df_vol):
        df = pd.merge(df_dates, df_spot, how="left", on="date")
        df = pd.merge(df, df_vol, how="left", on="date")
        df = df.replace('null', np.nan).fillna(method="ffill")
        df = df.dropna(axis=0, how='any').reset_index(drop=True)
        return df
    """
    def align(df_dates, df_spot, df_vol):
        df = df_dates.set_index('date').join(df_spot.set_index('date'), how='left')
        df = df.join(df_vol.set_index('date'), how='left')
        df.reset_index(level=0, inplace=True)
        df = df.replace('null', np.nan).fillna(method="ffill")
        df = df.dropna(axis=0, how='any').reset_index(drop=True)
        return df
    
    df = align(all_dates, df_spot, df_vol)
    df = prep_rv(df, asset_col, params)
    df = calcs_vol(df, params)

    this_df = df[["date", tenor_col, "blend"]]
    initial_startDate = params["first_date"]
    initial_decay = setup_decays(this_df, params, initial_startDate, tenor_col)

    tenor_compile_dates = pd.DataFrame(initial_startDate, index=range(1), columns=["StartDate"])  # Track all dates.
    
    #clean_decays = pd.merge(df[["date"]], initial_decay, how="left", on=["date"])  # Track all decays.
    clean_decays = df[["date"]].set_index('date').join(initial_decay.set_index('date'),how='left')
    clean_decays.reset_index(level=0, inplace=True)


    FLAG_PEAK = True
    last_startDate = initial_startDate
    last_startRow = df[df["date"] == last_startDate].index[0]
    
    #last_set = pd.merge(df[["date", tenor_col]], initial_decay[["date", "VT", "VB", "VH", "VHH"]], how="left", on="date")
    last_set = df.set_index('date').join(initial_decay[['date', "VT", "VB", "VH", "VHH"]].set_index('date'),how='left')
    last_set.reset_index(level=0, inplace=True)
    #print(last_set['date'])

    #return df, initial_decay, tenor_col, last_set

    while FLAG_PEAK:
        #rule_1_1 = last_set[tenor_col] > last_set[last_set["date"] == last_startDate][tenor_col].iloc[0]
        #print(last_startDate,last_set["date"].values)
        rule_1_1 = last_set[tenor_col] > last_set[tenor_col][np.where(last_set["date"].values == last_startDate)[0][0]]
        
        rule_1_1[0:last_startRow] = False
        rule_1_2 = last_set[tenor_col] > last_set["VH"]
        rule_1_2[0:last_startRow] = False
        rule_1 = rule_1_1 | rule_1_2

        regional_max = last_set[tenor_col].sort_index(ascending=0).rolling(window=int(params["days_nonewpeak"]), min_periods=1).max()
        regional_max = regional_max.sort_index(ascending=1)
        rule_3 = regional_max
        rule_3[0:last_startRow] = 0
        rule_3 = (rule_3 == last_set[tenor_col])

        rule_2_candidates = rule_1 & rule_3
        rule_2 = pd.Series(False, index=range(0, len(df)))
        if sum(rule_2_candidates) > 0:
            rule_2_index = list(rule_2_candidates[rule_2_candidates == 1].index)
            FLAG_TROUGH_EXIST = False
            this_peakCandt = 0

            while (not FLAG_TROUGH_EXIST) and (this_peakCandt <= (len(rule_2_index) - 1)):
                this_peak_row = rule_2_index[this_peakCandt]
                this_peak_iv = last_set.loc[this_peak_row, tenor_col]

                this_trough_list = (this_peak_iv - last_set[tenor_col]) / this_peak_iv
                this_trough_list[0:(this_peak_row + 1)] = 0
                this_trough_min = params["decline_pct"]
                this_trough_exist = (this_trough_list >= this_trough_min)
                if sum(this_trough_exist) > 0:
                    this_first_trough = this_trough_exist[this_trough_exist == 1].index[0]
                    this_peaks_check = [x for x in rule_2_index if (x > this_peak_row) and (x < this_first_trough)]
                    this_peaks_iv = last_set.loc[this_peaks_check, tenor_col]
                    if not any(x > this_peak_iv for x in this_peaks_iv):
                        FLAG_TROUGH_EXIST = True
                        rule_2[this_peak_row] = True
                    else:
                        this_peakCandt += 1
                else:
                    this_peakCandt += 1

            if FLAG_TROUGH_EXIST == True:
                this_startRow = this_peak_row
                this_df = df[["date", tenor_col, "blend"]]
                this_startDate = df.loc[this_startRow, "date"]
                this_decay = setup_decays(this_df, params, this_startDate, tenor_col)

                temp_compile_decays = pd.DataFrame(this_startDate, index=range(1), columns=["StartDate"])
                
                tenor_compile_dates = pd.concat([tenor_compile_dates, temp_compile_decays], axis=0, sort=False).reset_index(drop=True)
                #tenor_compile_dates = pd.DataFrame({tenor_compile_dates.columns[0]:np.append(tenor_compile_dates.values, temp_compile_decays.values)}) 
            
                this_decay = this_decay[this_decay["date"] >= this_startDate]
                clean_decays = clean_decays[clean_decays["date"] < this_startDate]
                clean_decays = pd.concat([clean_decays, this_decay], sort=False)

                last_startDate = this_startDate
                
                #last_startRow = df[df["date"] == last_startDate].index[0]
                #last_set = pd.merge(df[["date", tenor_col]], this_decay[["date", "VT", "VB", "VH", "VHH"]], how="left", on="date")
                last_startRow = np.where(df["date"].values == last_startDate)[0][0]
                last_set = df.set_index('date').join(this_decay[['date', "VT", "VB", "VH", "VHH"]].set_index('date'),how='left')
                last_set.reset_index(level=0, inplace=True)

            else:
                FLAG_PEAK = False
                break
        else:
            FLAG_PEAK = False
            break

    return clean_decays.reset_index(drop=True)


if __name__ == '__main__':
    a = pd.read_csv("a.csv")
    b = pd.read_csv("b.csv")
    params = {'blackswan_pct': 97.76345225422259,
              'days_nonewpeak': 10.0,
              'decline_pct': 0.20297994132697283,
              'high_vol_pctl': 43.50451160830091,
              'iv_pct': 25.59408324590975,
              'vb_decay': 0.16775183166705587,
              'vh_multiple': 14.954688097766748,
              'vhh_pctl': 33.85020626148733,
              'vlow_pctl': 35.9118028534882,
              'vt_decay': 0.16101442658595377,
              'vt_to_ly': 21.062077580010936,
              'weight_curr': 0.23561314974641812,
              'weight_rv': 0.4301857744186748,
              'first_date': '2004-05-14',
              'busdays': 21,
              'coeff': 12.0,
              'spot_start': '2001-01-01',
              'iv_start': '2004-04-08',
              'weight_short': 0.7643868502535819,
              'rv_pct': 74.40591675409024}

    start = t.perf_counter()
    c = decay_sets_generator(a, b, params)
    end = t.perf_counter()
    elapsed_time = end - start
    print(f"Elapsed Time: {elapsed_time:0.4f}")

    def wrapper(func, *args, **kwargs):
        def wrapped():
            return func(*args, **kwargs)
        return wrapped

    wrapped = wrapper(decay_sets_generator, a, b, params)
    print(timeit.timeit(wrapped,number=100)/100)
