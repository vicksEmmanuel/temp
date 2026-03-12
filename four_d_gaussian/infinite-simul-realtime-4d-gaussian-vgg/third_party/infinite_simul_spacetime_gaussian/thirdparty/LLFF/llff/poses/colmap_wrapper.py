import os
import subprocess



# $ DATASET_PATH=/path/to/dataset

# $ colmap feature_extractor \
#    --database_path $DATASET_PATH/database.db \
#    --image_path $DATASET_PATH/images

# $ colmap exhaustive_matcher \
#    --database_path $DATASET_PATH/database.db

# $ mkdir $DATASET_PATH/sparse

# $ colmap mapper \
#     --database_path $DATASET_PATH/database.db \
#     --image_path $DATASET_PATH/images \
#     --output_path $DATASET_PATH/sparse

# $ mkdir $DATASET_PATH/dense
def run_colmap(basedir, match_type):
    
    logfile_name = os.path.join(basedir, 'colmap_output.txt')
    logfile = open(logfile_name, 'w')
    
    feature_extractor_args = [
        'colmap', 'feature_extractor', 
            '--database_path', os.path.join(basedir, 'database.db'), 
            '--image_path', os.path.join(basedir, 'images'),
            '--ImageReader.single_camera', '1',
            '--SiftExtraction.use_gpu', '0',
    ]
    try:
        feat_output = ( subprocess.check_output(feature_extractor_args, universal_newlines=True) )
        logfile.write(feat_output)
    except subprocess.CalledProcessError as e:
        logfile.write(e.output)
        logfile.close()
        raise e
    print('Features extracted')

    exhaustive_matcher_args = [
        'colmap', match_type, 
            '--database_path', os.path.join(basedir, 'database.db'), 
            '--SiftMatching.use_gpu', '0',
    ]

    try:
        match_output = ( subprocess.check_output(exhaustive_matcher_args, universal_newlines=True) )
        logfile.write(match_output)
    except subprocess.CalledProcessError as e:
        logfile.write(e.output)
        logfile.close()
        raise e
    print('Features matched')
    
    p = os.path.join(basedir, 'sparse')
    if not os.path.exists(p):
        os.makedirs(p)

    mapper_args = [
        'colmap', 'mapper',
            '--database_path', os.path.join(basedir, 'database.db'),
            '--image_path', os.path.join(basedir, 'images'),
            '--output_path', os.path.join(basedir, 'sparse'),
            '--Mapper.num_threads', '16',
            '--Mapper.init_min_tri_angle', '4',
            '--Mapper.multiple_models', '0',
            '--Mapper.extract_colors', '0',
    ]

    try:
        map_output = ( subprocess.check_output(mapper_args, universal_newlines=True) )
        logfile.write(map_output)
    except subprocess.CalledProcessError as e:
        logfile.write(e.output)
        logfile.close()
        raise e
    logfile.close()
    print('Sparse map created')
    
    print( 'Finished running COLMAP, see {} for logs'.format(logfile_name) )


