#include "stdio.h"
#include "stdlib.h"
#include "string.h"

#include "hdf5.h"

#include "/Users/jra/prj/bqn/CBQN/include/bqnffi.h"


void check_status(herr_t status) {
    //fprintf(stderr, "status = %d\n", status);
    if (status < 0) {
        fprintf(stderr, "status = %d < 0; error; exiting\n", status);
        exit(EXIT_FAILURE);
    }
}


// readTrainX ← path_libreaddata •FFI "a"‿"read_train_x"‿""
BQNV read_train_x() {
    hid_t id_file;
    id_file = H5Fopen("datasets/train_catvnoncat.h5", H5F_ACC_RDONLY, H5P_DEFAULT);
    hid_t id_dataset;
    id_dataset = H5Dopen2(id_file, "/train_set_x", H5P_DEFAULT);
    herr_t status;
    uint8_t dataset[209 * 64 * 64 * 3];
    status = H5Dread(id_dataset, H5T_NATIVE_UINT8, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataset);
    check_status(status);
    BQNV res;
    res = bqn_makeC8Arr(4, (size_t[]){209, 64, 64, 3}, dataset);
    return res;
}


BQNV read_train_y() {
    hid_t id_file;
    id_file = H5Fopen("datasets/train_catvnoncat.h5", H5F_ACC_RDONLY, H5P_DEFAULT);
    hid_t id_dataset;
    id_dataset = H5Dopen2(id_file, "/train_set_y", H5P_DEFAULT);
    herr_t status;
    int64_t dataset[209];
    status = H5Dread(id_dataset, H5T_NATIVE_INT64, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataset);
    check_status(status);
    int8_t data[209];
    for (int i = 0; i < 209; i++) { data[i] = dataset[i]; }
    BQNV res;
    res = bqn_makeI8Vec(209, data);
    return res;
}


BQNV read_test_x() {
    hid_t id_file;
    id_file = H5Fopen("datasets/test_catvnoncat.h5", H5F_ACC_RDONLY, H5P_DEFAULT);
    hid_t id_dataset;
    id_dataset = H5Dopen2(id_file, "/test_set_x", H5P_DEFAULT);
    herr_t status;
    uint8_t dataset[209 * 64 * 64 * 3];
    status = H5Dread(id_dataset, H5T_NATIVE_UINT8, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataset);
    check_status(status);
    BQNV res;
    res = bqn_makeC8Arr(4, (size_t[]){209, 64, 64, 3}, dataset);
    return res;
}


BQNV read_test_y() {
    hid_t id_file;
    id_file = H5Fopen("datasets/test_catvnoncat.h5", H5F_ACC_RDONLY, H5P_DEFAULT);
    hid_t id_dataset;
    id_dataset = H5Dopen2(id_file, "/test_set_y", H5P_DEFAULT);
    herr_t status;
    int64_t dataset[209];
    status = H5Dread(id_dataset, H5T_NATIVE_INT64, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataset);
    check_status(status);
    int8_t data[209];
    for (int i = 0; i < 209; i++) { data[i] = dataset[i]; }
    BQNV res;
    res = bqn_makeI8Vec(209, data);
    return res;
}
