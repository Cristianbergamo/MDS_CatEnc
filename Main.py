# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 10:44:57 2017

@author: bergamo
"""
import pandas as pd
import numpy as np
import scipy.stats as sst
from sklearn.metrics import euclidean_distances
from sklearn import manifold
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
from pylab import figure
from sklearn.preprocessing import MinMaxScaler

np.random.seed(1)


class encoder():
    def __init__(self, dataset=None,
                 subject_column=None,
                 object_categorical_columns=None,
                 object_continuous_columns=None,
                 continuous_variables_methods=None,
                 output_columns=2
                 ):

        self.dataset = dataset
        self.subject_column = subject_column
        self.object_categorical_columns = object_categorical_columns
        self.object_continuous_columns = object_continuous_columns
        self.continuous_variables_methods = continuous_variables_methods
        self.output_columns = output_columns
        self.distance_matrix = None
        self.crosstable = None
        self.classes = None
        self.coding_dictionary = None

        if type(object_categorical_columns) != list:
            self.object_categorical_columns = [object_categorical_columns]
        if type(object_continuous_columns) != list:
            self.object_continuous_columns = [object_continuous_columns]
        if type(continuous_variables_methods) != list:
            self.continuous_variables_methods = [continuous_variables_methods]

    def get_cross_table(self):
        data = self.dataset
        subj = self.subject_column
        cat = self.object_categorical_columns
        cont = self.object_continuous_columns
        meth = self.continuous_variables_methods
        crosstable = None

        # object:categorical variables
        if not cat is None:
            flag_exist = 1
            crosstable = pd.crosstab(data[subj], data[cat[0]])
            if len(cat) > 1:
                for a in cat[1:]:
                    crosstable = pd.concat([crosstable,
                                            pd.crosstab(data[subj],
                                                        data[a])]
                                           , axis=1,
                                           join='outer')
            crosstable = crosstable * 0.5

        # object:continuous variables
        if not cont is None:

            if flag_exist == 0:
                crosstable = pd.pivot_table(data[[subj, cont[0]]],
                                            index=subj,
                                            aggfunc=self.get_aggr_func(meth[0]))

                if len(meth) > 1:
                    for m in meth[1:]:
                        crosstable = pd.concat([crosstable, (pd.pivot_table(data[[subj, cont[0]]],
                                                                            index=subj,
                                                                            aggfunc=self.get_aggr_func(m)))],
                                               axis=1,
                                               join='outer')

                        crosstable.rename(columns={crosstable.columns[-1]: crosstable.columns[-1] + '_' + m},
                                          inplace=True)



            else:
                crosstable = pd.concat([crosstable, (pd.pivot_table(data[[subj, cont[0]]],
                                                                    index=subj,
                                                                    aggfunc=self.get_aggr_func(meth[0])))],
                                       axis=1,
                                       join='outer')

                if len(meth) > 1:
                    for m in meth[1:]:
                        crosstable = pd.concat([crosstable, (pd.pivot_table(data[[subj, cont[0]]],
                                                                            index=subj,
                                                                            aggfunc=self.get_aggr_func(m)))],
                                               axis=1,
                                               join='outer')

                        crosstable.rename(columns={crosstable.columns[-1]: crosstable.columns[-1] + '_' + m},
                                          inplace=True)

            if len(cont) > 1:

                for a in cont[1:]:
                    crosstable = pd.concat([crosstable,
                                            pd.pivot_table(data[[subj, a]],
                                                           index=subj,
                                                           aggfunc=self.get_aggr_func(meth[0]))]
                                           , axis=1,
                                           join='outer')

                    if len(meth) > 1:
                        for m in meth[1:]:
                            crosstable = pd.concat([crosstable,
                                                    pd.pivot_table(data[[subj, a]],
                                                                   index=subj,
                                                                   aggfunc=self.get_aggr_func(m))]
                                                   , axis=1,
                                                   join='outer')

                            crosstable.rename(columns={a: a + '_' + m}, inplace=True)

            if len(cat) > 1:

                for a in cont[1:]:
                    crosstable = pd.concat([crosstable,
                                            pd.pivot_table(data[[subj, a]],
                                                           index=subj,
                                                           aggfunc=self.get_aggr_func(meth[0]))]
                                           , axis=1,
                                           join='outer')

                    if len(meth) > 1:
                        for m in meth[1:]:
                            crosstable = pd.concat([crosstable,
                                                    pd.pivot_table(data[[subj, a]],
                                                                   index=subj,
                                                                   aggfunc=self.get_aggr_func(m))]
                                                   , axis=1,
                                                   join='outer')

                            crosstable.rename(columns={a: a + '_' + m}, inplace=True)

        df_crosstable = crosstable.copy()
        mm = MinMaxScaler()
        crosstable = crosstable.fillna(0)
        crosstable = mm.fit_transform(crosstable)
        for i, a in enumerate(df_crosstable.columns):
            df_crosstable[a] = crosstable[:, i]

        crosstable = df_crosstable

        self.crosstable = crosstable
        self.classes = np.array(self.crosstable.index)

        return

    def get_aggr_func(self,
                      string):

        dic = {'mean': np.mean
            , 'mode': sst.mode
            , 'std': np.std
            , 'median': np.median
            , 'min': np.min
            , 'max': np.max}

        return dic[string]

    def get_euclidean_distance(self):

        self.distance_matrix = euclidean_distances(self.crosstable.fillna(
            0))  # per ogni variabile continua, se la classe della categorica da decodificare è assente, viene imposto il valore 0

        return

    def get_mds_data(self):

        mds = manifold.MDS(n_components=self.output_columns, max_iter=3000, eps=1e-9, random_state=1,
                           dissimilarity="precomputed")
        out = mds.fit(self.distance_matrix).embedding_

        dic = {}
        for i, a in enumerate(self.classes):
            dic[a] = out[i]

        self.coding_dictionary = dic

        return dic

    def encode_column(self):
        self.get_cross_table()
        self.get_euclidean_distance()

        return self.get_mds_data()


if __name__ == '__main__':

    file_name = 'heroes_information.csv'
    df = pd.read_csv(file_name, sep=';')
    cont_col = ['Height', 'Weight']
    df[cont_col] = df[cont_col].fillna(-99)
    mm = MinMaxScaler()

    df[cont_col] = mm.fit_transform(np.array(df[cont_col]))
    df1 = df.copy()

    cat_obj = ['Gender', 'Eye color', 'Race', 'Hair color',
               'Publisher', 'Skin color', 'Alignment']

    to_convert = ['name']
    for column in to_convert:
        encode_class = encoder(df, column, cat_obj, cont_col, ['mean', 'median', 'min', 'max', 'std'], 3)
        encoder_dictionary = encode_class.encode_column()
        df1[column] = df[column].map(encoder_dictionary)
        mapping = encoder_dictionary
        fig = figure()
        ax = Axes3D(fig)
        for hero in mapping.keys():
            ax.scatter(mapping[hero][0], mapping[hero][1], mapping[hero][2], color='b')
            ax.text(mapping[hero][0], mapping[hero][1], mapping[hero][2], '%s' % hero, size=8, zorder=1,
                    color='k')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        pyplot.show()

