class DataProcessor:
    # Constants for reading input data
    _PATH                = './data/'
    _DIGITS              = 5
    _OBJ_SUFFIX          = "obj"
    _RELATION_SUFFIX     = "relations"
    _FILE_TYPE           = ".txt"
    _LABELS              = ['type','id','x','y','z','w','h']
    FIXED_POSITIONS      = []
    OBJ_IDS_TO_EXCLUDE   = ['left','table']

    # Reads in data from a csv file and processes it in the required format
    def self.__init__(self, path):
        self._PATH = path

    def get_file_path(self, path, frame_no, digits, suffix, file_ending):
        # returns the file path with a specific file name format
        # e.g. in this case it is 00000_obj.txt
        fname = str(frame_no).zfill(digits)
        if len(suffix) == 0:
            suffix = ""
        else:
            suffix = "_" + suffix
        file_path = path + fname + suffix + file_ending
        return file_path

    def read_relations_from_csv(self, cycle):
        # reads in concept relations
        # assumes that each line consists of concept name and N variables
        # e.g. object-on-left,output_subassembly0,case0
        file_path = self.get_file_path(self._PATH, cycle, self._DIGITS, self._RELATION_SUFFIX, self._FILE_TYPE)
        lines    = open(file_path, "r")
        relations_dict = []
        for line in csv.reader(lines, delimiter=','):
            relations_dict += [[l for l in line]]
        return relations_dict

    def read_obj_data_from_csv(self, file_path):
        # assume that data_path points towards a csv file
        # where the format is 'objType,objId,x,y,z,w,h'
        lines    = open(file_path, "r")
        percepts = []
        for line in csv.reader(lines, delimiter=','):
            line = line[:len(_LABELS)]
            percepts.append([l if i <= 1 else float(l) for i,l in enumerate(line)])
    #       [objType,objId,x,y,z,w,h] = line
        return percepts

    def preattend(self, cycle):
        # Reads in data from CSV as a nested array
        # E.g. [['hand', 'hand2', '752', '214.5', '0', '124', '147'], ..]
        # Returns an array of concepts of type with initialised confidence value
        # [{'type': 'hand', id: 'hand2', 'x':752, ..}]

        file_path     = self.get_file_path(self._PATH, cycle, self._DIGITS, self._OBJ_SUFFIX, self._FILE_TYPE)
        percepts      = self.read_obj_data_from_csv(file_path)
        percepts_dict = [dict(zip(_LABELS,p)) for p in percepts]
        for i in percepts_dict:
            i.update({'conf': 0})
        return percepts_dict

