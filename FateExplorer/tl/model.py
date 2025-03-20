# -*- coding: utf-8 -*-
"""
Created on Fri Mar 14 14:57:26 2025

@author: jjy
"""

import warnings
warnings.filterwarnings('ignore')
import scanpy as sc
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from scipy.stats import spearmanr
from scipy.sparse import issparse

class Clone2Vec:
    
    def __init__(self,
                 mt,
                 ncores=10):
        self.mt = mt
        self.mt_filtered = None
        self.ncores = ncores
        self.adata = None
        self.contribution_key = None
        
    def embed(self,tokenize_method='spearman',
              clone_size_thr=5,cor_thr=0.2,dims=5,
                  window=5,rep_name = 'CloneEmbed',
                  epochs = 100,sg_method=1,umap_n_neighbors=20,
                  umap_min_dist = 0.5):
        
        adata = sc.AnnData(X=self.mt)
        mt_filtered = self.mt[self.mt.sum(axis=1) >= clone_size_thr]
        
        ### tokenize
        if tokenize_method=='spearman':
            correlation_matrix, _ = spearmanr(mt_filtered, axis=1)
        elif tokenize_method=='pearson':
            correlation_matrix = np.corrcoef(mt_filtered)
        
        correlation_df = pd.DataFrame(correlation_matrix, index=mt_filtered.index, 
                                      columns=mt_filtered.index)
        
        sentences = []
        retained_clones = []
        for idx, row in correlation_df.iterrows():
        
            related_clones = related_clones = row[(row > cor_thr) & (row.index != idx)].sort_values(ascending=False).index
            sentence = list(map(str, related_clones))
        
            if sentence:
                sentences.append(sentence)
                retained_clones.append(idx)
                
        retained_clones_str = [str(element) for element in retained_clones]
        filtered_adata = adata[adata.obs_names.isin(retained_clones_str)]
        
        ### train model
        model = Word2Vec(sentences, vector_size=dims, window=window, 
                         min_count=1, 
                         workers=self.ncores,sg=sg_method,epochs=epochs)
        
        clone_embeddings = np.array([
            np.mean([model.wv[word] for word in sentence], 
                    axis=0) if sentence else np.zeros(model.vector_size)
            for sentence in sentences
        ])
        filtered_adata.obsm[rep_name] = clone_embeddings
        self.mt_filtered = mt_filtered
        self.adata = filtered_adata
        sc.pp.neighbors(self.adata,use_rep=rep_name,
                        n_neighbors=umap_n_neighbors)
        sc.tl.umap(self.adata,min_dist=umap_min_dist)
        
    def add_clone_contribution(self,
                              normalize_method='log10'):
        if normalize_method == 'log10':
            
            mt_filtered = pd.DataFrame(self.mt_filtered)
            mt_filtered = mt_filtered.apply(lambda x: np.log10(x) if \
                                np.issubdtype(x.dtype, np.number) else x)
            mt_filtered.replace(-np.inf, 0, inplace=True)
            
        elif normalize_method == 'ratio':
            mt_filtered = self.mt_filtered.div(self.mt_filtered.sum(axis=1), 
                                               axis=0)
        
        key_list = []
        for col in list(mt_filtered.columns):
            key_list.append(col+'_'+normalize_method)
            self.adata.obs[col+'_'+normalize_method] = \
                mt_filtered.loc[self.adata.obs_names, col].values
        
    def aggregate_clone_feature(self,
                                adata,
                                clone_key='barcodes'):
        
        adata.obs['barcodes'] = adata.obs['barcodes'].astype(str)
        adata = adata[adata.obs.barcodes.isin(self.adata.obs_names)]
        barcodes = adata.obs[clone_key]

        if issparse(adata.X):
            data_matrix = adata.X.toarray()
        else:
            data_matrix = adata.X
    
        expression_df = pd.DataFrame(data_matrix, 
                                     index=adata.obs.index, 
                                     columns=adata.var_names)
        expression_df[clone_key] = barcodes.values
    
        # group expression
    
        clone_expression = expression_df.groupby(clone_key).mean()

    
        adata_final = sc.AnnData(X=clone_expression.values, 
                                 var=adata.var, 
                                 obs=pd.DataFrame(index=clone_expression.index))
        adata_final  = adata_final[self.adata.obs_names]
        # rewrite clone anndata
    
        for key in self.adata.obsm.keys():
            adata_final.obsm[key] = self.adata.obsm[key]
        adata_final.obs = self.adata.obs
        self.adata =  adata_final
        
    def plot_clone_contribution(self,
                                normalize_method='log10',**kwargs):
        self.celltype_index = [f"{col}_{normalize_method}" for col in self.mt.columns]
        sc.pl.umap(self.adata,color=self.celltype_index,**kwargs)

    def map_fate_cell_manifold(self,adata,key_name = 'fate_cluster'):
        
        if key_name not in adata.obs.columns:
            adata.obs[key_name] = None
            name_to_leiden = dict(zip(self.adata.obs_names, 
                                      self.adata.obs['leiden']))
        
        
        for idx, barcode in adata.obs['barcodes'].iteritems():
        
            if barcode in name_to_leiden:
                adata.obs.at[idx, key_name] = name_to_leiden[barcode]
                
        return adata

        
        
        
        
    
    
    
    
    
