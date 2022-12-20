import numpy as np

run_on_local = True
  # probably that 0 and 1 are irrelevant... not much data.

# import numpy as np
import pandas as pd
import datetime

# import modin.pandas as pd
# todo add csvs - to ssh

if run_on_local:
    path = "/Users/galbd/Desktop/clalit/csvs/icds/"
    #path = "/Users/galbd/Desktop/clalit/csvs/icds/need_to_upload/"
    #i = 0  # later - 0,1,2,3
    nrows_=20000
    loop_index = [1]
else:  # runs on ssh

    path = "input-icds/"
    nrows_=10000000000000000
    loop_index = [2, 3]

# from_year_=[1989,1996,2003,2010] # 5 years
from_year_ = [1987, 1994, 2001, 2008]  # 7 years
cut_year_ = [1993, 2000, 2007, 2014]
max_year_ = [1998, 2005, 2013, 2019]  # 5 years


def groupsize(dataframe_):
    df_size = dataframe_[['PID']]
    df_size = df_size.drop_duplicates(subset=['PID'], keep='first')
    return str(df_size.shape[0])


def replace_nans(df, value):

    if value=="mean":
        cols = df.columns
        for col in cols:
            if df[col].isnull().any():
                print(col)
                mean=df[col].mean()
                #print("contain nans")
                df[col] = df[col].fillna(mean)
    else:
        cols = df.columns
        for col in cols:
            if df[col].isnull().any():
                #print(col)
                #print("contain nans")
                df[col] = df[col].fillna(value)
    return df


# read all icds dataframe
for i in loop_index:
    print("loop index: " + str(i))
    from_year = from_year_[i]
    cut_year = cut_year_[i]
    max_year = max_year_[i]

    # todo dont forget the elimination of patients

    # <editor-fold desc="measurments">

    # dont have data before 1999 1999 ...

    measures = pd.read_csv(path + "dbo_Req783_Measures.csv")
    measures['EntryDate'] = pd.to_datetime(measures['Entry_Date'])
    measures['year'] = pd.DatetimeIndex(measures['EntryDate']).year
    measures_relevant = measures.loc[(measures['year'] >= from_year) & (measures['year'] <= cut_year)]
    measures_relevant = measures_relevant.drop(['Source'], axis=1)


    m_mean = measures_relevant.groupby(['PID', 'MeasureName']).mean()
    m_mean.reset_index(inplace=True)
    m_mean = m_mean.pivot(index='PID', columns='MeasureName', values='Result')
    m_mean.loc[m_mean['BMI']>60,'BMI']= 26.3
    m_mean.columns = ['BMI_mean', 'Heigth_mean', 'Weight_mean']

    m_median = measures_relevant.groupby(['PID', 'MeasureName']).median()
    m_median.reset_index(inplace=True)
    m_median = m_median.pivot(index='PID', columns='MeasureName', values='Result')
    m_median.loc[m_median['BMI']>60,'BMI']= 26.3
    m_median.columns = ['BMI_median', 'Heigth_median', 'Weight_median']

    m_max = measures_relevant.groupby(['PID', 'MeasureName']).max()
    m_max.reset_index(inplace=True)
    m_max = m_max.pivot(index='PID', columns='MeasureName', values='Result')
    m_max.loc[m_max['BMI']>60,'BMI']= 26.3
    m_max.columns = ['BMI_max', 'Heigth_max', 'Weight_max']

    m_last = measures_relevant.groupby(['PID', 'MeasureName']).last()
    m_last.reset_index(inplace=True)
    m_last = m_last.pivot(index='PID', columns='MeasureName', values='Result')
    m_last.loc[m_last['BMI']>60,'BMI']= 26.3
    m_last.columns = ['BMI_last', 'Heigth_last', 'Weight_last']

    measures_united = m_mean
    measures_united = measures_united.merge(m_median, how='left', on='PID')
    measures_united = measures_united.merge(m_max, how='left', on='PID')
    measures_united = measures_united.merge(m_last, how='left', on='PID')

    measures_united['obesity'] = measures_united['BMI_last'] > 30

    # </editor-fold>

    #measures_united

    # icds:

    # <editor-fold desc="read_icd (not mainpulating it..">
    df = pd.read_csv(path + "dbo_Req783_AllDiagnosisChameleon.csv", low_memory=False,nrows=nrows_)
    df['EntryDate'] = df['EventDate']
    df_ind = df[['PID', 'EntryDate', 'DiagnosisCode', 'DiagnosisDesc']]
    df_ind = df_ind.assign(Source='dbo_Req783_AllDiagnosisChameleon')

    df2 = pd.read_csv(path + "dbo_Req783_ChronicDiags_RawData.csv", low_memory=False,nrows=nrows_)
    df2_ind = df2[['PID', 'EntryDate', 'DiagnosisCode', 'DiagnosisDesc']]
    df2_ind = df2_ind.assign(Source='dbo_Req783_ChronicDiags_RawData')

    df3 = pd.read_csv(path + "dbo_Req783_DiagnosisChameleon_ER.csv", low_memory=False,nrows=nrows_)
    df3['EntryDate'] = df3['Admission_Date']
    df3_ind = df3[['PID', 'EntryDate', 'DiagnosisCode', 'DiagnosisDesc']]
    df3_ind = df3_ind.assign(Source='dbo_Req783_DiagnosisChameleon_ER')

    df4 = pd.read_csv(path + "dbo_Req783_DiagnosisChameleon_Hosp.csv", low_memory=False,nrows=nrows_)
    df4['EntryDate'] = df4['Open_Record_Date']
    df4_ind = df4[['PID', 'EntryDate', 'DiagnosisCode', 'DiagnosisDesc']]
    df4_ind = df4_ind.assign(Source='dbo_Req783_DiagnosisChameleon_Hosp')

    df5 = pd.read_csv(path + "dbo_Req783_DiagnosisChameleon_Visits.csv", low_memory=False,nrows=nrows_)
    df5['EntryDate'] = df5['EventDate']
    df5_ind = df5[['PID', 'EntryDate', 'DiagnosisCode', 'DiagnosisDesc']]
    df5 = df5.assign(Source='dbo_Req783_DiagnosisChameleon_Visits')

    df6 = pd.read_csv(path + "dbo_Req783_DiagnosisER_RelatedED.csv", low_memory=False,nrows=nrows_)
    df6['EntryDate'] = df6['ErEncounterDate']
    df6_ind = df6[['PID', 'EntryDate', 'DiagnosisCode', 'DiagnosisDesc']]
    df6 = df6.assign(Source='dbo_Req783_DiagnosisER_RelatedED')

    df7 = pd.read_csv(path + "dbo_Req783_DiagnosisHosp_RelatedED.csv", low_memory=False,nrows=nrows_)
    df7['EntryDate'] = df7['HospEncounterDate']
    df7_ind = df7[['PID', 'EntryDate', 'DiagnosisCode', 'DiagnosisDesc']]
    df7_ind = df7_ind.assign(Source="dbo_Req783_DiagnosisHosp_RelatedED")

    df8 = pd.read_csv(path + "dbo_Req783_DiagnosisOfek.csv", low_memory=False,nrows=nrows_)
    df8['EntryDate'] = df8['EventDate']
    df8_ind = df8[['PID', 'EntryDate', 'DiagnosisCode', 'DiagnosisDesc']]
    df8_ind = df8_ind.assign(Source='dbo_Req783_DiagnosisOfek')

    df9 = pd.read_csv(path + "dbo_Req783_ComplicationsDiagnosisOfek.csv", low_memory=False,nrows=nrows_)
    df9['EntryDate'] = df9['EventDate']
    df9_ind = df9[['PID', 'EntryDate', 'DiagnosisCode', 'DiagnosisDesc']]
    df9_ind = df9_ind.assign(Source='dbo_Req783_ComplicationsDiagnosisOfek')

    df10 = pd.read_csv(path + "dbo_Req783_IndxWithSource.csv", low_memory=False,nrows=nrows_)
    df10['EntryDate'] = df10['EventDate']
    df10_ind = df10[['PID', 'EntryDate', 'DiagnosisCode', 'DiagnosisDesc']]
    df10_ind = df10_ind.assign(Source='dbo_Req783_IndxWithSource')

    df11 = pd.read_csv(path + "dbo_Req783_DiagnosisGastroChameleonOld.csv", low_memory=False,nrows=nrows_)
    df11['EntryDate'] = df11['EventDate']
    df11['DiagnosisDesc'] = df11['Name']
    df11['DiagnosisCode'] = df11['Icd']
    df11_ind = df11[['PID', 'EntryDate', 'DiagnosisCode', 'DiagnosisDesc']]
    df11_ind = df11_ind.assign(Source='DiagnosisGastroChameleonOld')

    df12 = pd.read_csv(path + "dbo_Req783_Procedure.csv", low_memory=False,nrows=nrows_)
    df12['EntryDate'] = df12['EventDate']
    df12['DiagnosisCode'] = df12['ProcedureCode']
    df12['DiagnosisDesc'] = "Procedure_" + df12['ProcedureName']
    df12_ind = df12[['PID', 'EntryDate', 'DiagnosisCode', 'DiagnosisDesc']]
    df12_ind = df12_ind.assign(Source='dbo_Req783_Procedure')

    df13 = pd.read_csv(path + "dbo_Req783_EndicationFromProcedure.csv", low_memory=False,nrows=nrows_)
    df13['EntryDate'] = df13['EventDate']
    df13['DiagnosisCode'] = df13['Icd']
    df13['DiagnosisDesc'] = "Endication_from_Procedure_" + df13['Name']
    df13_ind = df13[['PID', 'EntryDate', 'DiagnosisCode', 'DiagnosisDesc']]
    df13_ind = df13_ind.dropna()
    df13_ind = df13_ind.assign(Source='dbo_Req783_EndicationFromProcedure')

    df_all_ind = pd.concat([df_ind, df2_ind, df3_ind, df4_ind, df5_ind, df6_ind, df7_ind, df8_ind, df9_ind, df10_ind, df11_ind, df12_ind,df13_ind])
    #df_all_ind['DiagnosisCode'] = df_all_ind['DiagnosisCode'].astype(str)
    #df_all_ind.DiagnosisCode = df_all_ind.DiagnosisCode.astype('string')
    df_all_ind['DiagnosisCode'] = df_all_ind['DiagnosisCode'].str.replace(' ', '',regex=True)
    df_all_ind['DiagnosisCode'] = df_all_ind['DiagnosisCode'].str.replace('.', '',regex=True)

    df_all_ind['EntryDate'] = pd.to_datetime(df_all_ind['EntryDate'])
    df_all_ind['year'] = pd.DatetimeIndex(df_all_ind['EntryDate']).year

    icds_ind = df_all_ind.loc[(df_all_ind['year'] >= from_year) & (df_all_ind['year'] <= cut_year)]
    classification_ind = df_all_ind.loc[(df_all_ind['year'] <= max_year) & (df_all_ind['year'] > cut_year)]
    # </editor-fold>

    # blood and lab tests
    # todo ADD BMI and into CLASSIFICATION as needed

    # <editor-fold desc=“icds”>

    print("all data frame group size is: " + groupsize(icds_ind))
    # clinical groups

    reA = '571[ .]*[598]'
    reB = '250'
    reC = '401'
    reD = '272[ .]*[0]| 272[ .]*[1]| 272[ .]*[2]| 272[ .]*[3]| 272[ .]*[4]'
    reE = '278[ .]0'
    reF = '571[ .]*[0]| 571[ .]*[1]| 571[ .]*[2]| 571[ .]*[3]| 571[ .]*[4]| 571[ .]*[6]|070|303'

    icds_ind_for_nafld = icds_ind[icds_ind['DiagnosisCode'].str.match(reA) == False]

    groupA = classification_ind[classification_ind['DiagnosisCode'].str.match(reA) == True]
    groupA = groupA[['PID', 'EntryDate', 'DiagnosisCode']]
    print("group A group size is: " + groupsize(groupA))

    groupB = classification_ind[classification_ind['DiagnosisCode'].str.match(reB) == True]
    groupB = groupB[['PID', 'EntryDate', 'DiagnosisCode']]
    print("group B group size is: " + groupsize(groupB))

    groupC = classification_ind[classification_ind['DiagnosisCode'].str.match(reC) == True]
    groupC = groupC[['PID', 'EntryDate', 'DiagnosisCode']]
    print("group C group size is: " + groupsize(groupC))

    groupD = classification_ind[classification_ind['DiagnosisCode'].str.match(reD) == True]
    groupD = groupD[['PID', 'EntryDate', 'DiagnosisCode']]
    print("group D group size is: " + groupsize(groupD))

    groupE1 = classification_ind[classification_ind['DiagnosisCode'].str.match(reE) == True]
    groupE1 = groupE1[['PID', 'EntryDate', 'DiagnosisCode']]


    groupE2 = measures_united[measures_united['obesity'] == True]
    groupE2.reset_index(inplace=True)
    groupE2 = groupE2[['PID']]
    groupE2.assign(EntryDate=cut_year)
    groupE2.assign(DiagnosisCode="BMI>30")

    groupE = pd.concat([groupE1, groupE2])

    print("group E group size is: " + groupsize(groupE))

    groupF = classification_ind[classification_ind['DiagnosisCode'].str.match(reF) == True]
    groupF = groupF[['PID', 'EntryDate', 'DiagnosisCode']]
    print("group D group size is: " + groupsize(groupD))

    # NAFLD : (GROUP A) - (GROUP F) - (OTHER LIVER DISEASES INDICATION)

    # remove group f from group A

    NAFLD_join = groupA.merge(groupF, on='PID', how='left', indicator=True)
    NAFLD = NAFLD_join[NAFLD_join['_merge'] == 'left_only']
    NAFLD = NAFLD.rename(columns={'EntryDate_x': 'EntryDate', 'DiagnosisCode_x': 'DiagnosisCode'})
    NAFLD = NAFLD[['PID', 'EntryDate', 'DiagnosisCode']]
    NAFLD = NAFLD.drop_duplicates(subset=['PID'], keep='first')
    NAFLD_ind = NAFLD[['PID', 'DiagnosisCode']]
    print("group NAFLD group size is: " + groupsize(NAFLD_ind))

    # todo need to remove also (OTHER LIVER DISEASES INDICATION)

    # PRESUMED NAFLD : {(GROUP A) or [(GROUP B) AND (GROUP C) AND (GROUP D)] or [(GROUP B) AND (GROUP E)]}
    #                    - (GROUP F) - (OTHER LIVER DISEASES INDICATION)

    groupB_C_join = groupB.merge(groupC, on='PID', how='inner', indicator=True)
    groupB_C_join = groupB_C_join.rename(columns={'EntryDate_x': 'EntryDate', 'DiagnosisCode_x': 'DiagnosisCode'})
    groupB_C = groupB_C_join[['PID', 'EntryDate', 'DiagnosisCode']]
    print("group BC group size is: " + groupsize(groupB_C))

    groupB_C_D_join = groupB_C.merge(groupD, on='PID', how='inner', indicator=True)
    groupB_C_D_join = groupB_C_D_join.rename(columns={'EntryDate_x': 'EntryDate', 'DiagnosisCode_x': 'DiagnosisCode'})
    groupB_C_D = groupB_C_D_join[['PID', 'EntryDate', 'DiagnosisCode']]
    print("group BCD group size is: " + groupsize(groupB_C_D))

    groupB_E_join = groupB.merge(groupE, on='PID', how='inner', indicator=True)
    groupB_E_join = groupB_E_join.rename(columns={'EntryDate_x': 'EntryDate', 'DiagnosisCode_x': 'DiagnosisCode'})
    groupB_E = groupB_E_join[['PID', 'EntryDate', 'DiagnosisCode']]
    print("group BE group size is: " + groupsize(groupB_E))

    PRESUMED_NAFLD = pd.concat([groupA, groupB_C_D, groupB_E])
    PRESUMED_NAFLD_join = PRESUMED_NAFLD.merge(groupF, on='PID', how='left', indicator=True)
    PRESUMED_NAFLD = PRESUMED_NAFLD_join[PRESUMED_NAFLD_join['_merge'] == 'left_only']
    PRESUMED_NAFLD = PRESUMED_NAFLD.rename(columns={'EntryDate_x': 'EntryDate', 'DiagnosisCode_x': 'DiagnosisCode'})
    PRESUMED_NAFLD = PRESUMED_NAFLD[['PID', 'EntryDate', 'DiagnosisCode']]
    PRESUMED_NAFLD = PRESUMED_NAFLD.drop_duplicates(subset=['PID'], keep='first')
    PRESUMED_NAFLD_ind = PRESUMED_NAFLD[['PID', 'DiagnosisCode']]
    print("group PRESUMED_NAFLD group size is: " + groupsize(PRESUMED_NAFLD_ind))

    # todo need to remove also (OTHER LIVER DISEASES INDICATION)
    print("1")

    # icds_ind_for_nafld.iloc[:,"DiagnosisCode"] = icds_ind_for_nafld.iloc[:,"DiagnosisCode"].apply(lambda x: str(x).strip())

    icds_ind_for_nafld = icds_ind_for_nafld.astype(str)
    # icds_ind_for_nafld['DiagnosisCode'] =icds_ind_for_nafld['DiagnosisCode'].replace(' ','')
    icds_ind_for_nafld['DiagnosisCode'] = icds_ind_for_nafld['DiagnosisCode'].replace(' ', "")
    icds_ind_for_nafld['DiagnosisCode'] = icds_ind_for_nafld['DiagnosisCode'].replace('.', "")
    # icds_ind_for_nafld=icds_ind_for_nafld.astype(str)
    icds_ind_for_nafld = icds_ind_for_nafld.drop_duplicates(subset=['PID', 'DiagnosisCode'], keep='first')
    icds_ind_for_nafld_ = icds_ind_for_nafld[['PID', 'DiagnosisCode', 'EntryDate']]

    df_all_ind_relevant_piv = icds_ind_for_nafld_.pivot_table(index='PID', columns='DiagnosisCode', fill_value=0,
                                                              aggfunc='count')
    # todo check if this create 2 column for each icd.. i think it is...
    df_all_ind_relevant_piv = df_all_ind_relevant_piv.droplevel(0, axis=1)


    dropped_columns = []

    for col in df_all_ind_relevant_piv.columns[1:]:
        if (df_all_ind_relevant_piv[col].sum()) < 90:  #
            dropped_columns.append(col)
    df_all_ind_relevant_piv.drop(columns=dropped_columns, inplace=True)
    print(df_all_ind_relevant_piv)


    df_all_ind_relevant_piv.reset_index(inplace=True)
    df_all_ind_relevant_piv['PID'] = df_all_ind_relevant_piv['PID'].astype(str)

    PRESUMED_NAFLD_ind = PRESUMED_NAFLD_ind.rename(columns={'DiagnosisCode': 'PRESUMED_NAFLD'})
    NAFLD_ind = NAFLD_ind.rename(columns={'DiagnosisCode': 'NAFLD'})

    PRESUMED_NAFLD_ind['PID'] = PRESUMED_NAFLD_ind['PID'].astype(str)
    NAFLD_ind['PID'] = NAFLD_ind['PID'].astype(str)

    df_icds = df_all_ind_relevant_piv.merge(PRESUMED_NAFLD_ind, on=['PID'], how='left')
    df_icds = df_icds.merge(NAFLD_ind, on='PID', how='left')
    df_icds.fillna(0, inplace=True)

    for col in df_icds.columns[1:]:
        df_icds[col] = df_icds[col].astype(bool)
    df_icds = df_icds.astype(int)
    df_icds.fillna(0, inplace=True)
    check=df_icds
    measures_united=replace_nans(measures_united,"mean")
    dataframe = measures_united.merge(df_icds, on='PID', how='left')
    dataframe.fillna(0, inplace=True)
    print("measures+df-icds:")

    d=dataframe

    print("***")
    print("")
    print("total number of patients:")
    count_all = df_icds['PID'].count()
    print(count_all)
    print("nafld:")
    print(df_icds['NAFLD'].sum())
    print(df_icds['NAFLD'].sum() / count_all)

    print("presumed:")
    print(df_icds['PRESUMED_NAFLD'].sum())
    print(df_icds['PRESUMED_NAFLD'].sum() / count_all)

    # for now - holding df_icds as input with both nafld and presumed indication..

    # </editor-fold>

    #    next featuers need to be joind with df_icds

    '''
    #demogrphics
    '''

    # <editor-fold desc="demogrphic">

    # demographics

    print("start demographics")

    demographic = pd.read_csv(path + "dbo_Req783_StaticDemographyWithPatientsFromBO.csv",nrows=nrows_)

    demographic['BirthDate'] = pd.to_datetime(demographic['BirthDate'])
    demographic['DeathDate'] = pd.to_datetime(demographic['DeathDate'])
    demographic['birth_year'] = pd.DatetimeIndex(demographic['BirthDate']).year
    demographic1 = demographic[pd.isna(demographic['DeathDate'])]
    cut_year_datetime = datetime.datetime.strptime(str(cut_year), '%Y')
    demographic2 = demographic[demographic['DeathDate'] > cut_year_datetime]
    demographic = pd.concat([demographic1, demographic2])
    new_demographic = demographic[['PID', 'GenderName','birth_year', 'OriginName', 'ImmigrationYear', 'SectorName']]
    new_demographic = pd.get_dummies(new_demographic, columns=['GenderName', 'OriginName', 'SectorName'])

    columns_to_drop = []
    for col in new_demographic.columns[6:]:
        if new_demographic[col].sum() < 100:
            columns_to_drop.append(col)

    new_demographic = new_demographic.drop(columns=columns_to_drop)
    # </editor-fold>

    # print(new_demographic)
    dataframe = dataframe.merge(new_demographic, on='PID', how='left')
    dataframe = replace_nans(dataframe, 0)
    untill_demo=dataframe
    '''
    #social state
    '''
    # <editor-fold desc="social state">
    social_state = pd.read_csv(path + "dbo_Req783_BO_SocialState_RawData.csv")
    social_state["ClinicUpdateDate"] = pd.to_datetime(social_state["ClinicUpdateDate"])
    social_state = social_state.sort_values(by="ClinicUpdateDate")
    social_state['year'] = pd.DatetimeIndex(social_state['ClinicUpdateDate']).year

    social_state_gb_first = social_state.groupby(['PID']).first()
    social_state_gb_last = social_state.groupby(['PID']).last()
    social_state_gb_first = social_state_gb_first[['SocialStateScore', 'ClinicSocialScore']]
    social_state_gb_last = social_state_gb_last[['SocialStateScore', 'ClinicSocialScore']]

    social_state_united = pd.merge(social_state_gb_first, social_state_gb_last, how="left", on="PID")
    social_state_united = social_state_united.rename(
        columns={"SocialStateScore_x": "SocialStateScore_first", "ClinicSocialScore_x": "ClinicSocialScore_first",
                 "SocialStateScore_y": "SocialStateScore_last", "ClinicSocialScore_y": "ClinicSocialScore_last"})
    # </editor-fold>
    social_state_united=replace_nans(social_state_united,"?")
    dataframe = dataframe.merge(social_state_united, on='PID', how='left')
    dataframe = replace_nans(dataframe, -1)
    dataframe=pd.get_dummies(dataframe)
    untill_social_state=dataframe
    '''
    #habits
    '''
    # <editor-fold desc="habits">
    HabitsChameleon = pd.read_csv(path + "dbo_Req783_HabitsChameleon.csv")
    HabitsChameleon["Entry_Date"] = pd.to_datetime(HabitsChameleon["Entry_Date"])
    HabitsChameleon_gb_first = HabitsChameleon.groupby(['PID']).first()
    HabitsChameleon_gb_first.reset_index(inplace=True)
    new_HabitsChameleon_gb_first = HabitsChameleon_gb_first[
        ['PID', 'Smoking', 'Smoking_Per_Day', 'Alcohol', 'Drugs']]
    new_HabitsChameleon_gb_first = new_HabitsChameleon_gb_first.rename(
        columns={ "Smoking": "is_smoking_first",
                 "Smoking_Per_Day": "Smoking_Per_Day_first", "Alcohol": "Alcohol_use_first",
                 "Drugs": "Drugs_use_first"})
    new_HabitsChameleon_gb_first = pd.get_dummies(new_HabitsChameleon_gb_first,
                                                  columns=['is_smoking_first', 'Alcohol_use_first', 'Drugs_use_first'])

    HabitsChameleon_gb_last = HabitsChameleon.groupby(['PID']).last()
    HabitsChameleon_gb_last.reset_index(inplace=True)
    new_HabitsChameleon_gb_last = HabitsChameleon_gb_last[
        ['PID', 'Smoking', 'Smoking_Per_Day', 'Alcohol', 'Drugs']]
    new_HabitsChameleon_gb_last = new_HabitsChameleon_gb_last.rename(
        columns={ "Smoking": "is_smoking_last",
                 "Smoking_Per_Day": "Smoking_Per_Day_last", "Alcohol": "Alcohol_use_last", "Drugs": "Drugs_use_last"})
    new_HabitsChameleon_gb_last = pd.get_dummies(new_HabitsChameleon_gb_last,
                                                 columns=['is_smoking_last', 'Alcohol_use_last', 'Drugs_use_last'])
    habits = new_HabitsChameleon_gb_first.merge(new_HabitsChameleon_gb_last, how='left', on='PID')
    #habits=replace_nans(habits,"mean")
    habits = habits.rename(
        columns={"SocialStateScore_x": "SocialStateScore_first", "ClinicSocialScore_x": "ClinicSocialScore_first",
                 "SocialStateScore_y": "SocialStateScore_last", "ClinicSocialScore_y": "ClinicSocialScore_last"})
    # </editor-fold>

    # habits
    dataframe = dataframe.merge(habits, on='PID', how='left')
    with_habits=dataframe
    dataframe = replace_nans(dataframe, -1)
    '''
     #lab tests
     '''
    # <editor-fold desc="lab_test- manipulate numeric - save not numeric">

    blood_tests = pd.read_csv(path + "all_blood_test.csv", low_memory=False,nrows=nrows_)
    blood_tests = blood_tests[['PID', 'EventDate', 'ind', 'test', 'res', 'lo', 'hi']]
    blood_tests['test'] = blood_tests['test'].str.replace(' ', '',regex=True)

    blood_tests["EventDate"] = pd.to_datetime(blood_tests["EventDate"])
    blood_tests["year"] = pd.DatetimeIndex(blood_tests['EventDate']).year
    blood_tests["year"] = blood_tests["year"].astype(int)

    blood_tests = blood_tests[blood_tests["year"] <= cut_year]
    blood_tests = blood_tests[blood_tests["year"] >= from_year]
    blood_tests = blood_tests.drop(columns=["year"])

    lst_numerric = []
    lst_not_numeric = []
    for index, row in blood_tests.iterrows():
        try:
            float(row["hi"])
            float(row["res"])
            float(row["lo"])
            lst_numerric.append(index)
        except:
            lst_not_numeric.append(index)

    df_numeric = blood_tests.loc[lst_numerric]
    # df_numeric.to_csv(path+"blood_test_relevant_numeric_"+str(i)+".csv")

    df_not_numeric = blood_tests.loc[lst_not_numeric]
    # df_not_numeric.to_csv(path+"blood_test_relevant_not_numeric_"+str(i)+".csv")

    blood_tests = 0

    # not numeric
    # <editor-fold desc="not_numeric">
    # region not_numeric
    df = df_not_numeric
    can = "ֿבוטל|נכון|מתוקנת|מספיק|לבדיקה|חומר|פעמים|מדבקה|לחזור|יש|טכניות|קודמת|זיהוי|לפי בקשת|הסליחה|טעות|עקב|פסל|קטנה|ֿאינו|פסול|בעבודה|קיימת|ריקה|נפסל|קודמת|שנית|לבדוק|לא|מבוטל"

    canceld = df[df['res'].str.contains(can) == True]
    not_canceld = df[df['res'].str.contains(can) == False]

    neg = "שלילי|קטן|Negative|NEGATIVE|negative"
    negative = not_canceld[not_canceld['res'].str.contains(neg) == True]
    prob_not_negative = not_canceld[not_canceld['res'].str.contains(neg) == False]
    small = "<"
    smaller = prob_not_negative[prob_not_negative['res'].str.contains(small) == True]
    not_negative = prob_not_negative[prob_not_negative['res'].str.contains(small) == False]

    negative_overall = pd.concat([negative, smaller])
    # not_negative=not_canceld[not_canceld['res'].str.contains(neg)==False]

    # pos="אעאעאע|ֿחיובי |גדול|Positive|POSITIVE|positive"
    pos = "חיובי|positive|POSITIVE|Positive"
    positive = not_negative[not_negative['res'].str.contains(pos) == True]
    prob_nor_postive_nor_negative = not_negative[not_negative['res'].str.contains(pos) == False]

    big = ">"
    bigger = prob_nor_postive_nor_negative[prob_nor_postive_nor_negative['res'].str.contains(big) == True]
    nor_postive_nor_negative = prob_nor_postive_nor_negative[
        prob_nor_postive_nor_negative['res'].str.contains(big) == False]

    positive_overall = pd.concat([positive, bigger])

    lst_numerric_new = []
    lst_not_numeric_new = []
    for index, row in nor_postive_nor_negative.iterrows():
        try:
            float(row["res"])
            lst_numerric_new.append(index)
        except:
            lst_not_numeric_new.append(index)

    # this does nothing..
    shit_happens = nor_postive_nor_negative.loc[lst_not_numeric_new]
    maybe_numeric = nor_postive_nor_negative.loc[lst_numerric_new]
    df_numeric_new = pd.concat([df_numeric, maybe_numeric])

    # negative_overall
    df_negative = negative_overall.groupby(['PID', 'test'])['res'].last()
    df_negative = df_negative.to_frame()
    df_negative.loc[:, 'res'] = True
    df_negative.reset_index(inplace=True)
    df_negative['test'] = (df_negative['test']).astype(str)
    df_negative['test'] = (df_negative['test']) + '_negative'
    df_negative_pivot = df_negative.pivot(index='PID', columns='test', values='res')
    df_negative_pivot = df_negative_pivot.fillna(False)

    # positive_overall

    df_positive = positive_overall.groupby(['PID', 'test'])['res'].last()
    df_positive = df_positive.to_frame()
    df_positive.loc[:, 'res'] = True
    df_positive.reset_index(inplace=True)
    df_positive['test'] = (df_positive['test']).astype(str)
    df_positive['test'] = (df_positive['test']) + '_positive'
    df_positive_pivot = df_positive.pivot(index='PID', columns='test', values='res')
    df_positive_pivot = df_positive_pivot.fillna(False)


    df_negative_pivot.reset_index(inplace=True)
    print(df_negative_pivot.columns)
    df_positive_pivot.reset_index(inplace=True)
    print(df_positive_pivot.columns)
    results = df_negative_pivot.merge(df_positive_pivot, how='outer', on='PID')
    # </editor-fold>
    # endregion

    columns_not_numeric = results.columns
    not_numeric = results
    [not_numeric[col].fillna(False, inplace=True) for col in columns_not_numeric]

    dataframe = dataframe.merge(not_numeric, on='PID', how='left')


    df = df_numeric_new
    df['test'] = df['test'].str.replace(' ', '')
    df['res'] = pd.to_numeric(df['res'], errors='coerce')
    df['hi'] = pd.to_numeric(df['hi'], errors='coerce')
    df['lo'] = pd.to_numeric(df['lo'], errors='coerce')

    df_mean = df.groupby(['PID', 'test'])['res'].mean()
    df_mean = df_mean.to_frame()
    df_mean.reset_index(inplace=True)
    df_mean['test'] = (df_mean['test']).astype(str)
    df_mean['test'] = (df_mean['test'])+ '_mean'
    df_mean_pivot = df_mean.pivot(index='PID', columns='test', values='res')

    df_median = df.groupby(['PID', 'test'])['res'].median()
    df_median = df_median.to_frame()
    df_median.reset_index(inplace=True)
    df_median['test'] = (df_median['test']).astype(str)
    df_median['test'] = (df_median['test']) + '_median'
    df_median_pivot = df_median.pivot(index='PID', columns='test', values='res')

    df_last = df.groupby(['PID', 'test'])['res'].last()
    df_last = df_last.to_frame()
    df_last.reset_index(inplace=True)
    df_last['test'] = (df_last['test']).astype(str)
    df_last['test'] = (df_last['test']) + '_last'
    df_last_pivot = df_last.pivot(index='PID', columns='test', values='res')

    df_first = df.groupby(['PID', 'test'])['res'].first()
    df_first = df_first.to_frame()
    df_first.reset_index(inplace=True)
    df_first['test'] = (df_first['test']).astype(str)
    df_first['test'] = (df_first['test']) + '_first'
    df_first_pivot = df_first.pivot(index='PID', columns='test', values='res')

    df_max = df.groupby(['PID', 'test'])['res'].max()
    df_max = df_max.to_frame()
    df_max.reset_index(inplace=True)
    df_max['test'] = (df_max['test']).astype(str)
    df_max['test'] = (df_max['test']) + '_max'
    df_max_pivot = df_max.pivot(index='PID', columns='test', values='res')

    df_min = df.groupby(['PID', 'test'])['res'].min()
    df_min = df_min.to_frame()
    df_min.reset_index(inplace=True)
    df_min['test'] = (df_min['test']).astype(str)
    df_min['test'] = (df_min['test']) + '_min'
    df_min_pivot = df_min.pivot(index='PID', columns='test', values='res')

    # place in scale
    sensitivity = 0.02

    df['lo'] = (df['lo']).astype(float)
    df['hi'] = (df['hi']).astype(float)
    print(df['hi'].head(30))
    print(df['lo'].head(30))
    df['min_max_norm_res'] = (df['res'] - df['lo']) / (df['hi'] - df['lo'])  # meanV,mediamV,lastV,firstV,maxV, minV
    df['min_max_norm_res_out_of_scale_up'] = df['min_max_norm_res'] > 1 + sensitivity  # sumV,lastV, meanV
    df['min_max_norm_res_out_of_scale_down'] = df['min_max_norm_res'] < -sensitivity  # sum itV, lastV, meanV
    df['min_max_norm_res_extreme'] = (df['min_max_norm_res'] - 0.5) ** 9  # last,median,mean, max, min

    df_min_max_norm_res_out_of_scale_up_sum = df.groupby(['PID', 'test'])['min_max_norm_res_out_of_scale_up'].sum()
    df_min_max_norm_res_out_of_scale_up_sum = df_min_max_norm_res_out_of_scale_up_sum.to_frame()
    df_min_max_norm_res_out_of_scale_up_sum.reset_index(inplace=True)
    df_min_max_norm_res_out_of_scale_up_sum['test'] = (df_min_max_norm_res_out_of_scale_up_sum['test']).astype(str)
    df_min_max_norm_res_out_of_scale_up_sum['test'] = (df_min_max_norm_res_out_of_scale_up_sum[
        'test']) + '_min_max_norm_sum'
    df_min_max_norm_res_out_of_scale_up_sum_pivot = df_min_max_norm_res_out_of_scale_up_sum.pivot(index='PID',
                                                                                                  columns='test',
                                                                                                  values='min_max_norm_res_out_of_scale_up')

    df_min_max_norm_res_out_of_scale_up_last = df.groupby(['PID', 'test'])['min_max_norm_res_out_of_scale_up'].last()
    df_min_max_norm_res_out_of_scale_up_last = df_min_max_norm_res_out_of_scale_up_last.to_frame()
    df_min_max_norm_res_out_of_scale_up_last.reset_index(inplace=True)
    df_min_max_norm_res_out_of_scale_up_last['test'] = (df_min_max_norm_res_out_of_scale_up_last['test']).astype(str)
    df_min_max_norm_res_out_of_scale_up_last['test'] = (df_min_max_norm_res_out_of_scale_up_last[
        'test']) + '_min_max_norm_last'
    df_min_max_norm_res_out_of_scale_up_last_pivot = df_min_max_norm_res_out_of_scale_up_last.pivot(index='PID',
                                                                                                    columns='test',
                                                                                                    values='min_max_norm_res_out_of_scale_up')

    df_min_max_norm_res_out_of_scale_up_mean = df.groupby(['PID', 'test'])['min_max_norm_res_out_of_scale_up'].mean()
    df_min_max_norm_res_out_of_scale_up_mean = df_min_max_norm_res_out_of_scale_up_mean.to_frame()
    df_min_max_norm_res_out_of_scale_up_mean.reset_index(inplace=True)
    df_min_max_norm_res_out_of_scale_up_mean['test'] = (df_min_max_norm_res_out_of_scale_up_mean['test']).astype(str)
    df_min_max_norm_res_out_of_scale_up_mean['test'] = (df_min_max_norm_res_out_of_scale_up_mean[
        'test']) + '_min_max_norm_mean'
    df_min_max_norm_res_out_of_scale_up_mean_pivot = df_min_max_norm_res_out_of_scale_up_mean.pivot(index='PID',
                                                                                                    columns='test',
                                                                                                    values='min_max_norm_res_out_of_scale_up')

    min_max_norm_res_out_of_scale_down_sum = df.groupby(['PID', 'test'])['min_max_norm_res_out_of_scale_down'].sum()
    min_max_norm_res_out_of_scale_down_sum = min_max_norm_res_out_of_scale_down_sum.to_frame()
    min_max_norm_res_out_of_scale_down_sum.reset_index(inplace=True)
    min_max_norm_res_out_of_scale_down_sum['test'] = (min_max_norm_res_out_of_scale_down_sum['test']).astype(str)
    min_max_norm_res_out_of_scale_down_sum['test'] = (min_max_norm_res_out_of_scale_down_sum[
        'test']) + '_min_max_norm_sum'
    min_max_norm_res_out_of_scale_down_sum_pivot = min_max_norm_res_out_of_scale_down_sum.pivot(index='PID',
                                                                                                columns='test',
                                                                                                values='min_max_norm_res_out_of_scale_down')

    min_max_norm_res_out_of_scale_down_last = df.groupby(['PID', 'test'])['min_max_norm_res_out_of_scale_down'].last()
    min_max_norm_res_out_of_scale_down_last = min_max_norm_res_out_of_scale_down_last.to_frame()
    min_max_norm_res_out_of_scale_down_last.reset_index(inplace=True)
    min_max_norm_res_out_of_scale_down_last['test'] = (min_max_norm_res_out_of_scale_down_last['test']).astype(str)
    min_max_norm_res_out_of_scale_down_last['test'] = (min_max_norm_res_out_of_scale_down_last[
        'test']) + '_min_max_norm_last'
    min_max_norm_res_out_of_scale_down_last_pivot = min_max_norm_res_out_of_scale_down_last.pivot(index='PID',
                                                                                                  columns='test',
                                                                                                  values='min_max_norm_res_out_of_scale_down')

    min_max_norm_res_out_of_scale_down_mean = df.groupby(['PID', 'test'])['min_max_norm_res_out_of_scale_down'].mean()
    min_max_norm_res_out_of_scale_down_mean = min_max_norm_res_out_of_scale_down_mean.to_frame()
    min_max_norm_res_out_of_scale_down_mean.reset_index(inplace=True)
    min_max_norm_res_out_of_scale_down_mean['test'] = (min_max_norm_res_out_of_scale_down_mean['test']).astype(str)
    min_max_norm_res_out_of_scale_down_mean['test'] = (min_max_norm_res_out_of_scale_down_mean[
        'test']) + '_min_max_norm_mean'
    min_max_norm_res_out_of_scale_down_mean_pivot = min_max_norm_res_out_of_scale_down_mean.pivot(index='PID',
                                                                                                  columns='test',
                                                                                                  values='min_max_norm_res_out_of_scale_down')
    df_merged = df_mean_pivot
    df_merged = df_merged.merge(df_last_pivot, how='left', on='PID')
    df_merged = df_merged.merge(df_max_pivot, how='left', on='PID')
    df_merged = df_merged.merge(df_min_pivot, how='left', on='PID')

    df_merged = df_merged.merge(df_min_max_norm_res_out_of_scale_up_sum_pivot, how='left', on='PID')
    df_merged = df_merged.merge(df_min_max_norm_res_out_of_scale_up_last_pivot, how='left', on='PID')
    df_merged = df_merged.merge(df_min_max_norm_res_out_of_scale_up_mean_pivot, how='left', on='PID')


    dataframe = dataframe.merge(df_merged, on='PID', how='left')
    #dataframe = replace_nans(dataframe, "mean")
    # </editor-fold>

    # <editor-fold desc="imaging and gastro-icds and meds">

    df_imaging = pd.read_csv(path + "dbo_Req783_AllImaging.csv", low_memory=False,nrows=nrows_)
    df_imaging['EntryDate'] = df_imaging['EventDate']
    # df_imaging['DiagnosisCode']=df_imaging['ProcedureNo']
    df_imaging = df_imaging[['PID', 'EntryDate', 'ProcedureNo', 'ImgProcedureTypeName', 'Result']]
    df_imaging['Result'] = df_imaging['Result']
    df_imaging_good = df_imaging[df_imaging['Result'].str.match("ללא|תקין|תקינה|סדיר") == True]
    df_imaging_good = df_imaging_good.assign(Imaging_is_problematic=-1)
    df_imaging_dont_know = df_imaging[df_imaging['Result'].str.match("ללא|תקין|תקינה|סדיר") == False]
    df_imaging_dont_know = df_imaging_dont_know.assign(Imaging_is_problematic=0.5)

    df_imaging_new = pd.concat([df_imaging_good, df_imaging_dont_know])
    df_imaging_new = df_imaging_new[['PID', 'EntryDate', 'ImgProcedureTypeName', 'Imaging_is_problematic']]
    #
    df_imaging_new["EntryDate"] = pd.to_datetime(df_imaging_new["EntryDate"])
    df_imaging_new["year"] = pd.DatetimeIndex(df_imaging_new['EntryDate']).year
    df_imaging_new["year"] = df_imaging_new["year"].astype(int)
    #
    df_imaging_new = df_imaging_new[df_imaging_new["year"] <= cut_year]
    df_imaging_new = df_imaging_new[df_imaging_new["year"] >= from_year]
    df_imaging_new = df_imaging_new.drop(columns=["year"])
    # df_imaging_new_last=df_imaging_new.groupby(['PID','ProcedureNo'])['Imaging_is_good'].last()

    df_imaging_new_min = df_imaging_new.groupby(['PID', 'ImgProcedureTypeName']).min()
    df_imaging_new_min = df_imaging_new_min.reset_index()
    df_imaging_new_min['ImgProcedureTypeName'] = df_imaging_new_min['ImgProcedureTypeName'].replace(' ', "")
    df_imaging_new_min_dd = df_imaging_new_min.drop_duplicates(subset=['PID', 'ImgProcedureTypeName'], keep='last')
    df_imaging_new_min_dd = df_imaging_new_min_dd.drop(columns=['EntryDate'])
    df_imaging_new_piv = df_imaging_new_min_dd.pivot_table(index='PID', columns='ImgProcedureTypeName', fill_value=0,
                                                           aggfunc='last')
    df_imaging_new_piv_ = df_imaging_new_piv.droplevel(0, axis=1)

    dataframe = dataframe.merge(df_imaging_new_piv_, how='left', on='PID')
    dataframe = replace_nans(dataframe, 0)
    #
    #
    # df_imaging_new_last_pivot=df_imaging_new.pivot(index=['PID','ProcedureNo'], columns='ProcedureNo', values='Imaging_is_good')
    #

    #
    #
    gastro_icds = pd.read_csv(path + "dbo_Req783_DuodenumGastroChameleonOld.csv", low_memory=False,nrows=nrows_)
    gastro_icds = gastro_icds.drop(
        columns=["Id_Num", "Patient", "Row_ID", "Hospital", "Unit", "Medical_Record", "Entry_Date", "Delete_Date",
                 "Delete_User", "Id_Num", "Patient", "Row_ID", "Hospital", "Unit", "Medical_Record", "Entry_Date",
                 "Entry_User", "Dynamic_Record_Date"])

    gastro_icds["EventDate"] = pd.to_datetime(gastro_icds["EventDate"])
    gastro_icds["year"] = pd.DatetimeIndex(gastro_icds['EventDate']).year
    gastro_icds["year"] = gastro_icds["year"].astype(int)
    #
    gastro_icds = gastro_icds[gastro_icds["year"] <= cut_year]
    gastro_icds = gastro_icds[gastro_icds["year"] >= from_year]
    gastro_icds = gastro_icds.drop(columns=["year"])

    gastro_icds_last_pivot = gastro_icds.groupby(['PID']).sum()
    b=dataframe
    dataframe = dataframe.merge(gastro_icds_last_pivot, how='left', on='PID')

    # todo read csvs
    med1 = pd.read_csv(path + "dbo_Req783_ChronicMedsER_RelatedED.csv")
    med1 = med1[["PID", "EntryDate", "ATC", "DrugName"]]

    med2 = pd.read_csv(path + "dbo_Req783_ChronicMedsHosp_RelatedED.csv")
    med2 = med2[["PID", "EntryDate", "ATC", "DrugName"]]

    med3 = pd.read_csv(path + "dbo_Req783_MedsBOBeforeEventDate.csv")
    med3 = med3[["PID", "EntryDate", "PharmacologicalCodeLevel_5", "DrugName"]]
    med3 = med3.rename(columns={"PharmacologicalCodeLevel_5": "ATC"})

    med4 = pd.read_csv(path + "dbo_Req783_MedsChameleonBeforeEventDate.csv")
    med4 = med4[["PID", "EventDate", "ATC", "Generic_Name_ForDisplay"]]
    med4 = med4.rename(columns={"EventDate": "EntryDate", "Generic_Name_ForDisplay": "DrugName"})

    med5 = pd.read_csv(path + "dbo_Req783_RegularMedsChameleon.csv")
    med5 = med5[["PID", "EntryDate", "ATC", "DrugName"]]

    med6 = pd.read_csv(path + "dbo_Req783_RegularMedsSpecificBO.csv")
    med6 = med6[["PID", "LastPrescriptionDate", "PharmacologicalCodeLevel_5", "DrugName"]]
    med6 = med6.rename(columns={"LastPrescriptionDate": "EntryDate", "PharmacologicalCodeLevel_5": "ATC"})

    meds_all = pd.concat([med1, med2, med3, med4, med5, med6])

    # todo continue from here :

    meds_all['EntryDate'] = pd.to_datetime(meds_all['EntryDate'])
    meds_all['year'] = pd.DatetimeIndex(meds_all['EntryDate']).year

    meds_all = meds_all.loc[(meds_all['year'] >= from_year) & (meds_all['year'] <= cut_year)]

    meds_all_ = meds_all.drop(columns=['year', 'EntryDate', 'ATC'])
    meds_all_ = meds_all.drop_duplicates(subset=['PID', 'ATC'], keep='last')
    meds_all_['ATC'] = meds_all_['ATC'].str.replace(' ', '',regex=True)

    meds_all_ = meds_all_.drop(columns=["EntryDate", "DrugName", "year"])
    meds_all_ = meds_all_.assign(ind=1)
    # todo change pivot to by ATC
    meds_all_piv = meds_all_.pivot_table(index='PID', columns='ATC', fill_value=0, aggfunc='last')
    meds_all_piv = meds_all_piv.droplevel(0, axis=1)

    dropped_columns = []
    for col in meds_all_piv.columns[1:]:
        if (meds_all_piv[col].sum()) < 55:  # 100
            print(col)
            dropped_columns.append(col)
    meds_all_piv.drop(columns=dropped_columns, inplace=True)

    dataframe = dataframe.merge(meds_all_piv, how='left', on='PID')
    dataframe = replace_nans(dataframe, 0)
    c=dataframe
    # </editor-fold>

    # <editor-fold desc="medications">
    # read csv
    # formatting
    # combine
    # filter by year
    # group by - last for each medication

    # </editor-fold>
    if run_on_local:
        dataframe.to_csv(path+"dataset_" + str(i) + "_.csv")
    else:
        dataframe.to_csv("outputs/dataset_" + str(i) + "_.csv")
    # this df holds all data for the patients besides his blood test results

# end of for loop
print("finish !! yes! ")
