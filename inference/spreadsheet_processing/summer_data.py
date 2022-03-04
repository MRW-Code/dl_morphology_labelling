import pandas as pd
import re
import numpy as np
import os


class SummerData():
    def __init__(self):
        self.df = pd.read_csv('../../data/summer_hts_data_matt.csv')
        self.clean_df = self.get_clean_df()
        self.fastai_df = self.get_fastai_df()

    def get_clean_df(self):
        df = self.df
        df = self.drop_extras(df)
        df = self.clean_eye_morphology(df)
        df = self.drop_non_morphology(df)
        df = self.get_image_paths(df)
        return df


    def drop_extras(self, df): return df.drop(['comments',
                                          'solubility_1',
                                          'solubility_2',
                                          'cas',
                                          'SMILES'], axis=1)

    def clean_eye_morphology(self, df):
        df['eye_morphology'].fillna('none', inplace=True)
        new_col = [re.findall(r'(\w*)', x)[0] for x in df['eye_morphology'].values]
        df['eye_morphology_clean'] = new_col
        df = df[df['eye_morphology_clean'] != 'none']
        return df


    def drop_non_morphology(self, df): return df.drop(['certainty',
                                                       'robot_folder',
                                                       'robot_morphology',
                                                       'crystalline',
                                                       'xray_folder',
                                                       'Unnamed: 0',
                                                       'image_quality',
                                                       'eye_morphology_old'], axis=1)

    def get_image_paths(self, df):
        fnames = []
        labels = []
        api = []
        for file in df['api']:
            base = f'../../data/water/water/{file}/'
            image = os.listdir(f'../../data/water/water/{file}')[0]
            fnames.append(f'{base}{image}')
            labels.append(self.df.eye_morphology[self.df['api'] == file].values[0])
        df['fname'] = fnames
        df['multi_label'] = labels
        new_df = df

            # for idx, _ in enumerate(os.listdir(f'../../data/water/water/{file}')):
            #     image = os.listdir(f'../../data/water/water/{file}')[idx]
            #     fnames.append(f'{base}{image}')
            #     labels.append(df.eye_morphology_clean[df['api'] == file].values[0])
            #     api.append(file)
            # new_df = pd.DataFrame({'fname' : fnames,
            #                    'eye_morphology_clean' : labels,
            #                    'api' : api})

        return new_df

    def get_fastai_df(self):
        # df = self.clean_df.drop('api', axis=1)
        df = self.clean_df
        df = df[df['eye_morphology_clean'] != 'rod']
        return df