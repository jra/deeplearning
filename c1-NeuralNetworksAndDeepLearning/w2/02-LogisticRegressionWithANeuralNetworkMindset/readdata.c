#include "stdio.h"
#include "stdlib.h"
#include "string.h"

#include "hdf5.h"

#include "/Users/jra/prj/bqn/CBQN/include/bqnffi.h"

#define PATHDS "/Users/jra/prj/coursera/deeplearning/c1-NeuralNetworksAndDeepLearning/w2/02-LogisticRegressionWithANeuralNetworkMindset/datasets"

void check_status(herr_t status) {
    //fprintf(stderr, "status = %d\n", status);
    if (status < 0) {
        fprintf(stderr, "status = %d < 0; error; exiting\n", status);
        exit(EXIT_FAILURE);
    }
}


uint8_t* read_from_hdf5(
    const char* path_file,
    const char* path_dataset,
    const size_t nbytes
    ) {
    hid_t id_file;
    id_file = H5Fopen(path_file, H5F_ACC_RDONLY, H5P_DEFAULT);
    hid_t id_dataset;
    id_dataset = H5Dopen2(id_file, path_dataset, H5P_DEFAULT);
    herr_t status;
    uint8_t* data = malloc(nbytes);
    status = H5Dread(id_dataset, H5T_NATIVE_UINT8, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
    check_status(status);
    return data;
}


BQNV read_train_x2() {
    uint8_t* dataset = (uint8_t*)read_from_hdf5(
        PATHDS "/train_catvnoncat.h5",
        "/train_set_x",
        209 * 64 * 64 * 3 * 1
        );
    BQNV res;
    res = bqn_makeC8Arr(4, (size_t[]){209, 64, 64, 3}, dataset);
    free(dataset);
    return res;
}


BQNV read_train_y2() {
    int64_t* dataset = (int64_t*)read_from_hdf5(
        PATHDS "/train_catvnoncat.h5",
        "/train_set_y",
        209 * 8
        );
    int8_t data[209];
    for (int i = 0; i < 209; i++) { data[i] = dataset[i]; }
    BQNV res;
    res = bqn_makeI8Vec(209, data);
    free(dataset);
    return res;
}


// readTrainX ← path_libreaddata •FFI "a"‿"read_train_x"‿""
BQNV read_train_x() {
    hid_t id_file;
    id_file = H5Fopen(PATHDS "/train_catvnoncat.h5", H5F_ACC_RDONLY, H5P_DEFAULT);
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
    id_file = H5Fopen(PATHDS "/train_catvnoncat.h5", H5F_ACC_RDONLY, H5P_DEFAULT);
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
    id_file = H5Fopen(PATHDS "/test_catvnoncat.h5", H5F_ACC_RDONLY, H5P_DEFAULT);
    hid_t id_dataset;
    id_dataset = H5Dopen2(id_file, "/test_set_x", H5P_DEFAULT);
    herr_t status;
    uint8_t dataset[50 * 64 * 64 * 3];
    status = H5Dread(id_dataset, H5T_NATIVE_UINT8, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataset);
    check_status(status);
    BQNV res;
    res = bqn_makeC8Arr(4, (size_t[]){50, 64, 64, 3}, dataset);
    return res;
}


BQNV read_test_y() {
    hid_t id_file;
    id_file = H5Fopen(PATHDS "/test_catvnoncat.h5", H5F_ACC_RDONLY, H5P_DEFAULT);
    hid_t id_dataset;
    id_dataset = H5Dopen2(id_file, "/test_set_y", H5P_DEFAULT);
    herr_t status;
    int64_t dataset[50];
    status = H5Dread(id_dataset, H5T_NATIVE_INT64, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataset);
    check_status(status);
    int8_t data[50];
    for (int i = 0; i < 50; i++) { data[i] = dataset[i]; }
    BQNV res;
    res = bqn_makeI8Vec(50, data);
    return res;
}


BQNV read_test_list_classes() {
    hid_t id_file;
    id_file = H5Fopen(PATHDS "/test_catvnoncat.h5", H5F_ACC_RDONLY, H5P_DEFAULT);
    hid_t id_dataset;
    id_dataset = H5Dopen2(id_file, "/list_classes", H5P_DEFAULT);
    herr_t status;
    //hid_t datatype = H5Dget_type(id_dataset);
    //hid_t class = H5Tget_class(datatype);
    //if (class == H5T_INTEGER) {
    //    fprintf(stderr, "read_test_list_classes class = %lld, H5T_INTEGER\n", class);
    //} else if (class == H5T_STRING) {
    //    fprintf(stderr, "read_test_list_classes class = %lld, H5T_STRING\n", class);
    //}
    hid_t id_type = H5Tcopy(H5T_C_S1);
    H5Tset_size(id_type, 7);
    H5Tset_strpad(id_type, H5T_STR_NULLPAD);
    //DATASET "list_classes" {
    //   DATATYPE  H5T_STRING {
    //      STRSIZE 7;
    //      STRPAD H5T_STR_NULLPAD;
    //      CSET H5T_CSET_ASCII;
    //      CTYPE H5T_C_S1;
    //   }
    //   DATASPACE  SIMPLE { ( 2 ) / ( 2 ) }
    //   DATA {
    //   (0): "non-cat", "cat\000\000\000\000"
    //   }
    //}
    uint8_t dataset[2 * 7];
    status = H5Dread(id_dataset, id_type, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataset);
    check_status(status);
    //fprintf(stderr, "read_test_list_classes >H5Dread dataset = >%s<\n", dataset);
    //for (int i = 0; i < 14; i++) {
    //    fprintf(stderr, "read_test_list_classes >H5Dread dataset[%d] = %d\n", i, dataset[i]);
    //}
    BQNV res;
    res = bqn_makeObjVec(2, (BQNV[]){
            bqn_makeC8Vec(7, &dataset[0]),
            bqn_makeC8Vec(3, &dataset[7]),
        });
    return res;
}
