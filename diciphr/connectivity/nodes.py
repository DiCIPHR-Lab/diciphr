import pandas as pd 
from ..resources import desikan86_nodes_csv
from ..utils import is_string

class Desikan86Nodes(object):
    __node_lookup = pd.read_csv(desikan86_nodes_csv)
    def __init__(self, labels=None, sizes=None, modules=None):
        self._read_from_csv()
        self.set_labels(labels=labels)
        self.set_sizes(sizes=sizes)
        self.set_modules(modules=modules)
    
    def __repr__(self):
        return self._dataframe.__repr__
        
    def __str__(self):
        return self._dataframe.__str__
    
    def _read_from_csv(self):
        self.X = tuple(self.__node_lookup['X'].astype(float))
        self.Y = tuple(self.__node_lookup['Y'].astype(float))
        self.Z = tuple(self.__node_lookup['Z'].astype(float))
        self.desikanlabels = tuple(self.__node_lookup['Label'].astype(int))
        self.numnodes = len(self.desikanlabels)
        self.shortnames = tuple(self.__node_lookup['ShortName'].astype(str))
        self.longnames = tuple(self.__node_lookup['LongName'].astype(str))
        self.shortnames2 = ('.'.join(_tup[::-1]) for _tup in (_el.split("_") for _el in self.shortnames))
        self.functionalsystem = tuple(self.__node_lookup['FunctionalSystem'].astype(str))
        self.functional_ids = tuple(self.__node_lookup['FunctionalID'].astype(int))
        self.cognitivesystem = tuple(self.__node_lookup['CognitiveSystem'].astype(str))
        self.lobe = tuple(self.__node_lookup['Lobe'].astype(str))
        self.lobe_ids = tuple(self.__node_lookup['LobeID'].astype(int))
        self.hemispheres = tuple(self.__node_lookup['Hemisphere'].astype(str))
        self.labels = list(self.desikanlabels)
        self.sizes = [1.0 for _i in range(self.numnodes)]
        self.modules = [1 for _i in range(self.numnodes)]
        
    def _update_dataframe(self):
        df = pd.DataFrame(index=range(self.numnodes), columns=['X','Y','Z','module','size','label'])
        df.loc[:,'X'] = self.X
        df.loc[:,'Y'] = self.Y
        df.loc[:,'Z'] = self.Z
        df.loc[:,'module'] = self.modules
        df.loc[:,'size'] = self.sizes
        df.loc[:,'label'] = self.labels
        self._dataframe = df
        
    def set_labels(self, labels=None):
        if labels is None:
            self.labels = list(self.desikanlabels)
        elif is_string(labels):
            if labels == 'range':
                self.labels = range(1,self.numnodes+1)
            elif labels.lower()[:5] == 'short':
                if labels[-1] == '2':
                    self.labels = list(self.shortnames2)
                else:
                    self.labels = list(self.shortnames)
            elif labels.lower()[:4] == 'long':
                self.labels = list(self.longnames)
        else:
            self.labels = list(labels)
        if not len(self.labels) == self.numnodes:
            raise ValueError('Must set labels with list of len {0}'.format(self.numnodes))
        self._update_dataframe()
    
    def set_sizes(self, sizes=None):
        if sizes is None:
            self.sizes = [1.0 for _i in range(self.numnodes)]
        elif isinstance(sizes,(int,float,long)):
            self.sizes = [float(sizes) for _i in range(self.numnodes)]
        else: 
            self.sizes = [float(size) for size in sizes]
        if not len(self.sizes) == self.numnodes:
            raise ValueError('Must set sizes with list of len {0}'.format(self.numnodes))
        self._update_dataframe()
            
    def set_modules(self, modules=None):
        if modules is None:
            self.modules = [1 for _i in range(self.numnodes)]
        elif is_string(modules):
            if modules.lower()[:4] == 'lobe':
                self.modules = list(self.lobe_ids)
            elif modules.lower()[:4] == 'func':
                self.modules = list(self.functional_ids)
            elif modules.lower()[:4] == 'hemi':
                self.modules = [1 if _el == 'L' else 2 for _el in self.hemispheres]
        else:
            self.modules = list(modules)
        if not len(self.modules) == self.numnodes:
            raise ValueError('Must set modules with list of len {0}'.format(self.numnodes))
        self._update_dataframe()
    
    def node_dataframe(self):
        return self._dataframe.copy()
        
    def to_file(self,filename):
        format_kwargs = { 
            'na_rep':'.', 
            'float_format':'%0.3f', 
            'header':False, 
            'index':False, 
            'sep':'\t'
        }
        self._dataframe.to_csv(filename, **format_kwargs)
