#include "relu_2_8_data.h"

ai_handle ai_relu_2_8_data_weights_get(void) {

    AI_ALIGNED(4)
    static const ai_u8 s_relu_2_8_weights[3504] = {
            0x80, 0xd3, 0x36, 0x43, 0x01, 0x91, 0xa4, 0x3e, 0x1e, 0x1c,
            0x94, 0x3e, 0xff, 0x07, 0x9a, 0x3e, 0x03, 0xbb, 0x61, 0x3e,
            0x28, 0x6f, 0x31, 0x3e, 0xcb, 0x2d, 0x6c, 0x40, 0x4d, 0x75,
            0xf1, 0x40, 0xc5, 0x14, 0x3a, 0xc3, 0x6c, 0x2f, 0xb0, 0xc3,
            0x65, 0x85, 0x8b, 0xc3, 0xef, 0x1e, 0x71, 0xc3, 0xc4, 0x28,
            0x9f, 0xc3, 0xa7, 0x58, 0x37, 0xc3, 0x9d, 0x57, 0x42, 0xc3,
            0xe7, 0x71, 0x0e, 0xc3, 0x10, 0xe4, 0x86, 0xbf, 0x73, 0x51,
            0x71, 0xbf, 0xc7, 0xda, 0x54, 0xbf, 0x56, 0x78, 0x44, 0xbf,
            0xc2, 0x95, 0x08, 0xbf, 0x3d, 0x5a, 0xe6, 0xbe, 0xc0, 0x20,
            0xae, 0xbe, 0x8d, 0xd4, 0x6b, 0xbe, 0xed, 0xce, 0xf6, 0xbd,
            0x12, 0x70, 0xd0, 0xbc, 0x9e, 0x60, 0xef, 0x3c, 0xb8, 0xe4,
            0xe1, 0x3d, 0xa0, 0xfd, 0x57, 0x3e, 0x50, 0xbc, 0xa5, 0x3e,
            0x4d, 0x19, 0xe0, 0x3e, 0x84, 0xd2, 0x10, 0x3f, 0xb9, 0x9a,
            0xb9, 0x9b, 0x99, 0x89, 0x9b, 0x8b, 0x8e, 0x88, 0xa7, 0xc6,
            0x99, 0xda, 0xb9, 0x98, 0x9a, 0x9a, 0x9a, 0x9a, 0xaa, 0x9a,
            0x99, 0xa9, 0xb7, 0xb7, 0x49, 0xb9, 0xa7, 0xf9, 0xa8, 0x87,
            0x9b, 0x67, 0x3c, 0x8a, 0xab, 0xaa, 0x99, 0xaa, 0xf9, 0xa7,
            0x6a, 0xcc, 0xbc, 0x76, 0x68, 0xc8, 0x99, 0x99, 0xaa, 0xaa,
            0xc6, 0xdc, 0xb8, 0xca, 0x88, 0x99, 0x9c, 0x89, 0x99, 0xa3,
            0x68, 0x98, 0xb9, 0xc9, 0x69, 0xa9, 0xa8, 0xca, 0x9c, 0xca,
            0xaa, 0x9a, 0x9a, 0x99, 0xa9, 0xa9, 0x9a, 0x99, 0x7a, 0xb4,
            0x4c, 0x96, 0xaa, 0x9a, 0xaa, 0x99, 0x7d, 0xbc, 0xeb, 0xaa,
            0xa9, 0xaa, 0x9a, 0xa9, 0xaa, 0xa9, 0xa9, 0x9a, 0xa9, 0xaa,
            0xaa, 0x99, 0xa9, 0xb8, 0xd8, 0x9a, 0xba, 0x8b, 0x9c, 0x7b,
            0xba, 0x85, 0x98, 0xab, 0xc7, 0x6c, 0x7f, 0xaa, 0x96, 0xcc,
            0xbd, 0x8b, 0xb8, 0xec, 0x6d, 0xc8, 0x9a, 0x9a, 0xaa, 0x99,
            0x9a, 0xab, 0xba, 0x99, 0x9a, 0xf8, 0x48, 0xc7, 0xa6, 0xb8,
            0x78, 0xc9, 0x99, 0x89, 0xba, 0xaa, 0xc5, 0xb6, 0x8d, 0xac,
            0xaa, 0x99, 0x9a, 0xa9, 0x9a, 0x89, 0xab, 0x9a, 0x89, 0xd4,
            0x87, 0x7b, 0x6d, 0xc8, 0xc8, 0xda, 0xa8, 0xc6, 0x88, 0xc9,
            0x7b, 0xb8, 0xb7, 0x88, 0xac, 0xbb, 0x99, 0xa9, 0x99, 0x9a,
            0xab, 0xaa, 0x76, 0x57, 0xcf, 0x8a, 0x9a, 0xa9, 0x99, 0x9a,
            0x9a, 0x89, 0xa9, 0x99, 0x99, 0xe9, 0xc7, 0x9b, 0x9a, 0xaa,
            0xaa, 0x99, 0x88, 0x87, 0xb9, 0xf7, 0xaa, 0xaa, 0xa9, 0x99,
            0xaa, 0xa6, 0x6d, 0x7a, 0x9b, 0xe4, 0x46, 0xcb, 0xa7, 0xd8,
            0xbc, 0xd7, 0x8e, 0x8c, 0x98, 0x99, 0x9f, 0xd7, 0x57, 0x67,
            0xdc, 0xac, 0x99, 0xda, 0x76, 0xd8, 0xca, 0x98, 0x9c, 0xbd,
            0xad, 0xab, 0xb9, 0xa8, 0xbc, 0xb8, 0x57, 0x70, 0x5b, 0xc9,
            0x99, 0xa9, 0x99, 0x99, 0x19, 0x02, 0xb1, 0x40, 0x2e, 0x5c,
            0xab, 0x40, 0xa9, 0x06, 0x8a, 0x40, 0xf3, 0xfa, 0x20, 0x40,
            0xd9, 0xd0, 0x2e, 0x40, 0x32, 0x44, 0x28, 0x40, 0xd0, 0x54,
            0xd1, 0x40, 0x05, 0x16, 0xdc, 0xbf, 0x10, 0x24, 0x89, 0xc0,
            0x14, 0x38, 0x67, 0x40, 0xb3, 0xf8, 0xd3, 0xc0, 0x24, 0x0c,
            0x11, 0x40, 0x93, 0xe0, 0x98, 0x40, 0xfe, 0x4a, 0x8c, 0x40,
            0x1c, 0xe1, 0xaf, 0x40, 0xe0, 0x9d, 0x94, 0x40, 0xe4, 0x94,
            0xe6, 0x40, 0x36, 0x7c, 0x27, 0xc1, 0x6b, 0x44, 0x59, 0x40,
            0x85, 0x09, 0x67, 0x40, 0xeb, 0x33, 0x4b, 0x40, 0xc5, 0x98,
            0x70, 0x40, 0x3b, 0x53, 0xc6, 0xbf, 0x1a, 0x15, 0x87, 0x40,
            0x9e, 0x03, 0x7d, 0x40, 0xd9, 0x7a, 0x23, 0x40, 0x05, 0xa7,
            0x67, 0x40, 0xf5, 0x38, 0x07, 0x41, 0xf2, 0x16, 0xa2, 0x40,
            0x39, 0xe7, 0x43, 0x40, 0x02, 0x68, 0x5a, 0x3f, 0x50, 0xb2,
            0xaf, 0x3f, 0xe3, 0x27, 0x53, 0x40, 0xf7, 0x20, 0x5d, 0x40,
            0x60, 0xed, 0x98, 0x3f, 0xd5, 0x15, 0xa4, 0x40, 0xe7, 0x45,
            0x71, 0x40, 0xf9, 0x4c, 0x60, 0x40, 0xc4, 0xea, 0x7b, 0x40,
            0x0b, 0xc8, 0x6e, 0x40, 0x2b, 0x18, 0xbf, 0x40, 0x45, 0x08,
            0x4f, 0x40, 0x66, 0x53, 0xfb, 0xc0, 0x0c, 0xf9, 0x70, 0x40,
            0xb3, 0xc3, 0xea, 0x40, 0x8c, 0xa8, 0x56, 0x40, 0x83, 0x71,
            0x71, 0xc0, 0x25, 0x8d, 0x54, 0x40, 0x4e, 0x02, 0x33, 0x40,
            0x54, 0x24, 0x63, 0x3f, 0xee, 0xcd, 0x45, 0x40, 0x25, 0x7d,
            0x1b, 0x3e, 0x46, 0x07, 0x4b, 0x40, 0x4c, 0xd1, 0x07, 0xc0,
            0x06, 0xd6, 0x2c, 0xc1, 0x9b, 0x60, 0x52, 0x3f, 0xde, 0x5a,
            0xa1, 0x40, 0xae, 0x8f, 0x4b, 0xc0, 0xc2, 0xf5, 0x0a, 0xc0,
            0xf3, 0x66, 0x80, 0x40, 0xc8, 0x6a, 0xd7, 0xbf, 0x20, 0x89,
            0x03, 0xc1, 0x13, 0x2f, 0x8b, 0xc0, 0xd5, 0x7c, 0x2f, 0x40,
            0x61, 0x61, 0x22, 0xc1, 0x71, 0x89, 0x14, 0xc1, 0x30, 0x81,
            0x09, 0xc1, 0x24, 0xb3, 0xf1, 0xc0, 0x6d, 0xcf, 0xd9, 0xc0,
            0x69, 0x6c, 0xb1, 0xc0, 0xe4, 0x3d, 0x95, 0xc0, 0x13, 0x34,
            0x74, 0xc0, 0x08, 0x1d, 0x15, 0xc0, 0xad, 0x80, 0xa7, 0xbf,
            0x49, 0x1b, 0x14, 0xbf, 0xad, 0x03, 0x2b, 0xbe, 0x1b, 0xa3,
            0x21, 0x3e, 0xb3, 0x3c, 0x07, 0x3f, 0xa6, 0x37, 0x84, 0x3f,
            0x85, 0x8e, 0xfc, 0x3f, 0xbc, 0xcc, 0xdd, 0xcc, 0xbc, 0xbc,
            0xcc, 0xbb, 0xbd, 0xdc, 0xdc, 0xbc, 0xde, 0xbb, 0xcc, 0xbd,
            0xec, 0xbc, 0xcd, 0xcc, 0xcc, 0xdc, 0xbc, 0xbb, 0xac, 0xcc,
            0xcb, 0xdc, 0xbc, 0xcb, 0xdd, 0xbe, 0xba, 0xcd, 0xcc, 0xdb,
            0xbb, 0xbc, 0xcb, 0xcc, 0xcb, 0xcc, 0xda, 0xda, 0xdc, 0xde,
            0xbb, 0xbb, 0xad, 0xdc, 0xac, 0xac, 0xcb, 0xae, 0xae, 0xcd,
            0xac, 0xcc, 0xbb, 0xab, 0xab, 0xcb, 0xbc, 0xcc, 0x0a, 0xcb,
            0xca, 0xba, 0xdc, 0xab, 0xb9, 0xbd, 0xba, 0xaa, 0xcd, 0xcd,
            0xba, 0xab, 0xbb, 0xdb, 0xad, 0xdb, 0xec, 0x9a, 0xcc, 0xbc,
            0xde, 0xb9, 0x9c, 0xbc, 0xbb, 0xbc, 0xbb, 0xbb, 0xbe, 0xbd,
            0xbd, 0xcb, 0xcb, 0xcb, 0xcb, 0xbc, 0xcc, 0xbc, 0xba, 0xbb,
            0xbb, 0xcb, 0xee, 0xdb, 0xcc, 0xdb, 0xce, 0xcb, 0xbc, 0xbc,
            0xac, 0xbb, 0xad, 0xbc, 0xcb, 0xbc, 0xdd, 0xac, 0xcd, 0xbc,
            0xbb, 0xdd, 0xcd, 0xbc, 0xee, 0xca, 0xbd, 0xbc, 0xcb, 0xcc,
            0xcd, 0xec, 0xbd, 0xbb, 0xcd, 0xbc, 0xbb, 0xbb, 0xcd, 0xbc,
            0xac, 0xdd, 0xcb, 0xbc, 0xcd, 0xdb, 0xcb, 0xdc, 0xcb, 0xbc,
            0xac, 0xcc, 0xca, 0xdd, 0xcb, 0xca, 0xab, 0xbc, 0xce, 0xcb,
            0xba, 0xad, 0xeb, 0xbc, 0xcb, 0xbb, 0xac, 0xcc, 0xbb, 0xdd,
            0xfb, 0xdb, 0xec, 0xae, 0xdc, 0xbc, 0xce, 0xbd, 0xab, 0xab,
            0xfb, 0xac, 0xbc, 0xbb, 0xca, 0xde, 0xde, 0xcc, 0xbd, 0xcb,
            0xbc, 0xca, 0xfa, 0xcc, 0xb9, 0xce, 0xbb, 0xbb, 0xdd, 0xbb,
            0xcc, 0xcb, 0xbb, 0xca, 0x9b, 0xcd, 0xbb, 0xbb, 0xbd, 0xee,
            0xbc, 0xcc, 0xda, 0xcb, 0xcd, 0xac, 0xdb, 0xbe, 0xcb, 0xcc,
            0xdc, 0xdb, 0xcb, 0xac, 0xeb, 0xdc, 0xdc, 0xce, 0xcc, 0xcb,
            0xcc, 0xbd, 0xcb, 0xcb, 0xdd, 0xdc, 0xdb, 0xbd, 0xdb, 0xdc,
            0xcc, 0xcd, 0xec, 0xdc, 0xdd, 0xcc, 0xcc, 0xcc, 0xdc, 0xcc,
            0xbe, 0xcc, 0xbb, 0xbb, 0xba, 0xcd, 0xec, 0xcc, 0xbc, 0xcc,
            0xdb, 0xbb, 0xce, 0xcc, 0xdb, 0xad, 0xcd, 0xac, 0xcd, 0xbd,
            0xcb, 0xac, 0xad, 0xbd, 0x9d, 0xdb, 0xdd, 0xac, 0xcc, 0xba,
            0xbb, 0xdc, 0xcd, 0xeb, 0xbf, 0xcc, 0xcc, 0xbb, 0xac, 0xac,
            0xbd, 0xbc, 0xca, 0xbb, 0xdd, 0xcb, 0xbc, 0xcc, 0xad, 0xcb,
            0xda, 0xad, 0xab, 0xbc, 0xba, 0xcb, 0xbd, 0xeb, 0xbb, 0xcc,
            0xac, 0xcc, 0xeb, 0xcc, 0xcd, 0xcd, 0xbb, 0xcb, 0xcc, 0xdd,
            0xbc, 0xbd, 0xdc, 0xbb, 0xda, 0xac, 0x9a, 0xdb, 0xab, 0xbe,
            0xdd, 0xdb, 0xcd, 0xbd, 0xdb, 0xeb, 0xac, 0xcc, 0xcc, 0xbc,
            0xcd, 0xeb, 0xbb, 0xba, 0xdd, 0xca, 0xab, 0xce, 0xcd, 0xcb,
            0xbe, 0xdc, 0xbc, 0xcc, 0xbc, 0xad, 0xcb, 0xbc, 0xdb, 0xdc,
            0xbb, 0xcc, 0xdd, 0xcd, 0xdb, 0xbe, 0xcb, 0xba, 0xbc, 0xbb,
            0xaa, 0xab, 0xeb, 0xbb, 0xbc, 0xce, 0xba, 0xbd, 0xbc, 0xcc,
            0xdd, 0xcc, 0xbd, 0xdb, 0xbe, 0xcc, 0xcc, 0xbc, 0xcd, 0xbb,
            0xdc, 0xcb, 0xcc, 0xcc, 0xdb, 0xcc, 0xac, 0xdd, 0xcc, 0xbc,
            0xcb, 0xba, 0xcc, 0xcd, 0xcc, 0xbb, 0xbb, 0xbc, 0xcb, 0xcc,
            0xda, 0xdb, 0xcd, 0xca, 0xdb, 0xab, 0xbb, 0xec, 0xbd, 0xed,
            0xac, 0xca, 0xcd, 0xdc, 0xbc, 0xcc, 0xdb, 0xcc, 0xec, 0xcc,
            0xbb, 0xca, 0xaa, 0xdd, 0xbb, 0xeb, 0xeb, 0xcb, 0xcd, 0xcd,
            0xaa, 0xcb, 0xce, 0xcd, 0xdd, 0xca, 0xce, 0xaa, 0xdf, 0xcb,
            0xea, 0xed, 0xcf, 0xc9, 0xed, 0xac, 0xbb, 0xdb, 0xcd, 0xba,
            0xdc, 0xca, 0xab, 0xac, 0xdd, 0xcd, 0xbb, 0xcc, 0xcb, 0xea,
            0xbb, 0xad, 0xbb, 0xcc, 0xbd, 0xcc, 0xcd, 0xbd, 0xbe, 0xac,
            0xfc, 0xaa, 0xdc, 0xfd, 0xcb, 0xca, 0xae, 0xbc, 0xbb, 0xbd,
            0xda, 0xbb, 0xcc, 0xec, 0xcc, 0xcc, 0xcb, 0xcf, 0xcc, 0xeb,
            0xec, 0xdc, 0xbb, 0xbc, 0xbb, 0xdf, 0xbd, 0xdc, 0xdd, 0xcc,
            0xbb, 0xdd, 0xfb, 0xbb, 0xbb, 0xce, 0xcb, 0xaa, 0xbe, 0xbe,
            0xdc, 0xcb, 0xeb, 0xbd, 0x9d, 0xcc, 0xcc, 0xbd, 0xbe, 0xbe,
            0xbc, 0xdb, 0xea, 0xbc, 0xab, 0xbb, 0xcb, 0xcd, 0xbc, 0xcb,
            0xce, 0xdc, 0xbc, 0xcb, 0xec, 0xbb, 0xda, 0xbd, 0xda, 0xbc,
            0xbd, 0xdb, 0xbb, 0xce, 0xcd, 0xba, 0xdc, 0xdd, 0xcc, 0xcc,
            0xbb, 0xbb, 0xbc, 0xdc, 0xcc, 0xbb, 0xbb, 0xba, 0xba, 0xbd,
            0xda, 0xbc, 0xbe, 0xcc, 0xcb, 0xed, 0xdc, 0xbb, 0xad, 0xba,
            0xbc, 0xba, 0xbe, 0xbd, 0xdc, 0xaa, 0x9d, 0xdc, 0xcb, 0xce,
            0xdc, 0xdd, 0xcd, 0xca, 0x9a, 0xeb, 0xea, 0xbb, 0xcb, 0xab,
            0xdb, 0xdd, 0xdd, 0xdb, 0xcd, 0xdc, 0xcc, 0xcd, 0xeb, 0xcc,
            0xdb, 0xca, 0xcc, 0xcc, 0xbb, 0xcd, 0xde, 0xbc, 0xcc, 0xdc,
            0xeb, 0xcd, 0xda, 0xdc, 0xbe, 0xdc, 0xec, 0xdb, 0xad, 0xcc,
            0xdd, 0xcc, 0xcc, 0xcd, 0xcb, 0xbc, 0xed, 0xbb, 0xca, 0xbc,
            0xbd, 0xcb, 0xcd, 0xdb, 0xba, 0xbb, 0xce, 0xcc, 0xcb, 0xbc,
            0xdd, 0xbb, 0xbb, 0xdc, 0xbb, 0xdc, 0xbc, 0xcc, 0xcc, 0xcb,
            0xdc, 0xcc, 0xcb, 0xcc, 0xdc, 0xcd, 0xbc, 0xbb, 0xdb, 0xcc,
            0xbb, 0xbc, 0xbc, 0xcb, 0xdb, 0xda, 0xbe, 0xbc, 0xe8, 0xbc,
            0xbb, 0xbc, 0xdc, 0xcb, 0xbc, 0xed, 0xbb, 0xbd, 0xde, 0xbe,
            0xcc, 0xdb, 0xcb, 0xcb, 0xbb, 0xbc, 0xcd, 0xbc, 0xac, 0xcc,
            0xcd, 0xcb, 0xcb, 0xdc, 0xcd, 0xbc, 0xca, 0xdb, 0xdd, 0xcc,
            0xcd, 0xdb, 0xdc, 0xcc, 0xcd, 0xcd, 0xdc, 0xdd, 0xcc, 0xcc,
            0xbc, 0xaa, 0xcc, 0xcc, 0xdb, 0xbc, 0xcc, 0xcb, 0xab, 0xcd,
            0xbc, 0xdc, 0xdc, 0xca, 0xcc, 0xcc, 0xbc, 0xbc, 0xce, 0xac,
            0xcc, 0xca, 0xca, 0xcb, 0xcb, 0xcb, 0xdb, 0xdb, 0xcd, 0xdb,
            0xcd, 0xcb, 0xbd, 0xbc, 0xbb, 0xcb, 0xec, 0xcb, 0xcc, 0xcc,
            0xac, 0xcd, 0xde, 0xdc, 0xec, 0xcc, 0xb9, 0xbc, 0xcb, 0xdc,
            0xee, 0xbc, 0xcb, 0xad, 0xea, 0xdb, 0xcb, 0xbc, 0xde, 0xab,
            0xbb, 0xdd, 0xcb, 0xcd, 0xab, 0xab, 0xac, 0x9b, 0xbc, 0xbb,
            0xcc, 0xcb, 0xcb, 0xbb, 0xcb, 0xbc, 0xcc, 0xbc, 0xcc, 0xdb,
            0xcd, 0xbc, 0xcd, 0xbd, 0xcb, 0xcb, 0xcd, 0xca, 0xbc, 0xcc,
            0xdb, 0xcc, 0xcc, 0xcb, 0xcc, 0xcc, 0xba, 0xcc, 0xcd, 0xcc,
            0xeb, 0xcc, 0xbb, 0xcb, 0xcc, 0xcc, 0xac, 0xcb, 0xdd, 0xcd,
            0xcb, 0xcb, 0xea, 0xbc, 0xcd, 0xbd, 0xdb, 0xbb, 0xdc, 0xcb,
            0xcb, 0xcc, 0xdc, 0xbc, 0xcd, 0xbc, 0xcc, 0xdd, 0xbc, 0xbc,
            0xbb, 0xdb, 0xdd, 0xcb, 0xbd, 0xdb, 0xbb, 0xce, 0xcc, 0xbb,
            0xb9, 0xcc, 0xba, 0xbc, 0xdc, 0xcb, 0xcb, 0xbb, 0xdd, 0xba,
            0xba, 0xcc, 0xcb, 0xbc, 0xbb, 0xcd, 0x8d, 0xac, 0xcc, 0xdc,
            0xcd, 0xba, 0xbc, 0xac, 0xbb, 0xcc, 0xcc, 0xbc, 0xcb, 0xda,
            0xad, 0xcc, 0xdb, 0xbc, 0xac, 0xbb, 0x9c, 0xaa, 0xbd, 0xee,
            0xbd, 0xcd, 0xcd, 0xbb, 0xbc, 0xda, 0xdb, 0xcd, 0xbc, 0x9d,
            0xcc, 0xbb, 0xcc, 0xcf, 0xeb, 0xbb, 0xca, 0xbb, 0xbb, 0xcb,
            0xab, 0xdd, 0xbd, 0xcc, 0xdc, 0xcd, 0xcd, 0xcb, 0xbc, 0xcc,
            0xbd, 0xcf, 0xcd, 0xba, 0xdb, 0xcb, 0xcc, 0xbd, 0xec, 0xcb,
            0xbc, 0xbe, 0xcb, 0xcd, 0xbc, 0xde, 0xbb, 0xcc, 0xed, 0xdb,
            0xbd, 0xba, 0xcc, 0xbc, 0x99, 0xbb, 0xfd, 0xbc, 0xba, 0xbd,
            0x9c, 0xbd, 0xbc, 0xbd, 0xba, 0xcc, 0xdc, 0x9d, 0xac, 0xab,
            0xcb, 0xcc, 0xbd, 0xca, 0xdc, 0xcc, 0xbb, 0xcc, 0xbd, 0x9a,
            0xba, 0xbd, 0xbb, 0xdb, 0xcb, 0xbb, 0xcc, 0xbc, 0xdd, 0xdc,
            0xcb, 0xbc, 0xbd, 0xbc, 0xdd, 0xac, 0xcc, 0xba, 0xec, 0xcc,
            0xba, 0xcd, 0xec, 0xcc, 0xdc, 0xbc, 0xbb, 0xcd, 0xcc, 0xbb,
            0xcc, 0xbb, 0xcb, 0xcc, 0xbb, 0xdb, 0xcc, 0xdd, 0xcb, 0xcd,
            0xdd, 0xcc, 0xcd, 0xbb, 0xec, 0xbc, 0xcd, 0xce, 0xcd, 0xdc,
            0xcd, 0xcc, 0xbc, 0xbd, 0xdb, 0xcc, 0xcc, 0xda, 0xcd, 0xdc,
            0xbb, 0xbe, 0xac, 0xdb, 0xeb, 0xcb, 0xbb, 0xbb, 0xcc, 0xcd,
            0xac, 0xba, 0xb9, 0xbe, 0xce, 0xcc, 0xdc, 0xdc, 0xcc, 0x9d,
            0xcd, 0xda, 0xcb, 0xcc, 0xcc, 0xbc, 0xfa, 0xcc, 0xcc, 0xcd,
            0xcb, 0xac, 0xca, 0xbc, 0x9b, 0xdc, 0xec, 0xcb, 0xbc, 0xba,
            0xab, 0xcd, 0xbd, 0xcc, 0xbd, 0xbc, 0xbc, 0xcc, 0xcb, 0xdc,
            0xbd, 0xca, 0xcb, 0xbc, 0xcb, 0xca, 0xcc, 0xbc, 0xbd, 0xbc,
            0xda, 0xad, 0xdc, 0xcc, 0xbd, 0xdb, 0xdc, 0xdc, 0xcb, 0xcc,
            0xbc, 0xcb, 0xbd, 0xce, 0xac, 0xdd, 0xdc, 0xbb, 0xcb, 0xcc,
            0xac, 0xcc, 0xcc, 0xec, 0xbb, 0xcd, 0xdc, 0xdb, 0xca, 0xcb,
            0xdc, 0xbc, 0xbc, 0xbb, 0xcb, 0xbd, 0xbc, 0xbe, 0xac, 0xda,
            0xcb, 0xbc, 0xcc, 0xcc, 0xbd, 0xdd, 0xbd, 0xcd, 0xdd, 0xac,
            0xbd, 0xab, 0xdd, 0xcc, 0xdb, 0xce, 0xcc, 0xbc, 0xcc, 0xcb,
            0xbc, 0xab, 0xeb, 0xdd, 0xdc, 0xcd, 0xcc, 0xcb, 0xbd, 0xac,
            0x9c, 0xdc, 0xdb, 0xbc, 0xac, 0xbc, 0xbc, 0xdd, 0xbb, 0xcd,
            0xdd, 0xbc, 0xcb, 0xcc, 0xbc, 0xbc, 0xdd, 0xbe, 0xcc, 0xcd,
            0xdd, 0xcc, 0xcc, 0xbd, 0xed, 0xdb, 0xeb, 0xbd, 0xcc, 0xdb,
            0xcc, 0xbc, 0xdc, 0xcd, 0xdc, 0xcb, 0xbc, 0xbb, 0xdc, 0xdc,
            0xdc, 0xbe, 0xed, 0xdb, 0xbd, 0xac, 0xbd, 0xba, 0xba, 0xdd,
            0xcb, 0xcc, 0xdd, 0xbc, 0xcc, 0xca, 0xcb, 0xdb, 0xab, 0xcd,
            0xcc, 0xbc, 0xcc, 0xcd, 0xba, 0xdb, 0xea, 0xcb, 0xca, 0xbc,
            0xbd, 0xce, 0xaa, 0xdc, 0xeb, 0xcb, 0xbe, 0xcb, 0xab, 0xcb,
            0xca, 0xdc, 0xcc, 0xba, 0xed, 0xbc, 0xdc, 0xcc, 0xbc, 0xcc,
            0xdb, 0xbc, 0xdb, 0xcb, 0xcb, 0xab, 0xdc, 0xdb, 0xdc, 0xdc,
            0xac, 0xbb, 0xbc, 0xcf, 0xbc, 0xbc, 0xfe, 0xcc, 0xbe, 0xec,
            0x9f, 0xec, 0xbe, 0xee, 0xab, 0xb9, 0xff, 0xea, 0xdb, 0xad,
            0xee, 0xbd, 0xad, 0xaf, 0xac, 0xac, 0xab, 0xce, 0xea, 0xec,
            0xfc, 0xbb, 0xbb, 0xcc, 0xaa, 0xbd, 0xad, 0xde, 0xed, 0xbb,
            0xcc, 0xab, 0xed, 0xbb, 0xca, 0xab, 0xce, 0xc9, 0xbc, 0xad,
            0xdc, 0xab, 0xac, 0xbb, 0xce, 0xeb, 0xcb, 0xda, 0xbb, 0xcd,
            0xcc, 0xdc, 0xdb, 0xad, 0xbc, 0xcd, 0xcc, 0xcd, 0xbb, 0xbb,
            0xdd, 0xda, 0xbd, 0xdc, 0xbc, 0xac, 0xaa, 0xbe, 0xbe, 0xda,
            0xcc, 0xcc, 0xdc, 0xbd, 0xec, 0xcb, 0xbc, 0xdd, 0xcb, 0xcc,
            0xbd, 0xcd, 0xad, 0xdc, 0xdd, 0xdb, 0xbc, 0xba, 0xbd, 0xbe,
            0xcc, 0xcc, 0xcc, 0xbc, 0xbc, 0xbc, 0xdc, 0xcc, 0xcb, 0xde,
            0xcc, 0xcd, 0xec, 0xbd, 0xba, 0xcc, 0xdb, 0xcc, 0xdd, 0xbc,
            0xbc, 0xcc, 0xbc, 0xad, 0xac, 0xca, 0xdb, 0xcc, 0xcb, 0xcc,
            0xcc, 0xcd, 0xdc, 0xdd, 0xdb, 0xbc, 0xcb, 0xcb, 0xbc, 0xdd,
            0xbc, 0xac, 0xbb, 0xba, 0xde, 0xdb, 0xcb, 0xad, 0xdc, 0xcb,
            0xec, 0xbe, 0xbc, 0xbb, 0xcc, 0xca, 0xdc, 0xac, 0xdc, 0xcc,
            0xbb, 0xcc, 0xdb, 0xc9, 0xcb, 0xbb, 0xeb, 0xcc, 0xba, 0xbc,
            0xcc, 0xdd, 0xcd, 0xca, 0xcb, 0xb9, 0xcb, 0xac, 0xdc, 0xbb,
            0xdc, 0xcb, 0xcc, 0xbd, 0xbc, 0xcd, 0xca, 0xab, 0xbc, 0xdc,
            0xdb, 0xdb, 0xcc, 0xcb, 0xbb, 0xbd, 0xbb, 0xdb, 0xdb, 0xdb,
            0xcb, 0xac, 0xdb, 0xdb, 0xbd, 0xeb, 0xcb, 0xba, 0xce, 0xed,
            0xac, 0xcc, 0xba, 0xcc, 0xac, 0xdb, 0xad, 0xac, 0xac, 0xad,
            0xed, 0x9b, 0xbb, 0xcc, 0xbb, 0xdb, 0xcb, 0xca, 0xbb, 0xcd,
            0xeb, 0xcb, 0xce, 0xbb, 0xac, 0x2b, 0xcc, 0xbe, 0xbb, 0xc9,
            0xda, 0xba, 0xdc, 0xcc, 0xbe, 0xcb, 0xac, 0xde, 0xbb, 0xbb,
            0xab, 0xbd, 0xeb, 0x8c, 0xbc, 0xbb, 0xbc, 0xcc, 0xbc, 0xce,
            0xca, 0xca, 0xcd, 0xdb, 0xbd, 0xcc, 0xed, 0xcc, 0xad, 0xde,
            0xba, 0xbb, 0xde, 0xcd, 0xab, 0xac, 0xfb, 0xcb, 0xeb, 0xdd,
            0xdb, 0xdb, 0xdd, 0xbe, 0x9b, 0xbb, 0xac, 0xab, 0xcb, 0xbc,
            0xca, 0xce, 0xdc, 0xdc, 0xbf, 0x9c, 0xba, 0xdb, 0xbb, 0xc9,
            0x7b, 0x9d, 0xdb, 0xcd, 0xcc, 0xcb, 0xcb, 0xcb, 0xbb, 0x9d,
            0xbc, 0xad, 0xbc, 0xbb, 0xdd, 0xca, 0xcc, 0xbd, 0xea, 0x7c,
            0xcc, 0xbb, 0xbb, 0x9f, 0xcc, 0xbd, 0xdd, 0xcb, 0xbb, 0xbc,
            0xdb, 0xbb, 0xbb, 0xec, 0xbd, 0xbc, 0xbe, 0xbd, 0xbb, 0xcc,
            0xdc, 0xbb, 0xbc, 0xec, 0xcb, 0xcc, 0xcd, 0xad, 0xbc, 0xdc,
            0xec, 0xbc, 0xbc, 0xbc, 0xca, 0xcd, 0xcb, 0xdc, 0xba, 0xbd,
            0xba, 0xcc, 0xeb, 0xcb, 0xab, 0xcd, 0xbc, 0xbb, 0xcc, 0xba,
            0xdc, 0xbc, 0xbb, 0xbd, 0xeb, 0xee, 0xcb, 0xac, 0xca, 0xce,
            0xac, 0xcb, 0xdc, 0xbb, 0xdb, 0xba, 0xbc, 0xdd, 0xdd, 0xd9,
            0x9c, 0xac, 0xbb, 0xca, 0xcc, 0xc9, 0x6d, 0xd9, 0xea, 0xbe,
            0xbe, 0xcb, 0xda, 0xb9, 0xad, 0xad, 0xbc, 0xfc, 0xac, 0xaa,
            0xeb, 0xca, 0xdd, 0xcd, 0x9b, 0x7c, 0xdd, 0xac, 0xca, 0x9b,
            0xbd, 0xdb, 0xcd, 0xdb, 0xcd, 0xbb, 0xdc, 0xbc, 0xbb, 0xce,
            0xbd, 0xab, 0xcd, 0xdd, 0xdd, 0xdc, 0xdc, 0xbc, 0xab, 0xec,
            0xbc, 0xbb, 0xcd, 0xcd, 0xdc, 0xcc, 0xec, 0xcc, 0xbb, 0xdd,
            0xbb, 0xbc, 0xac, 0xbb, 0xee, 0xbc, 0xce, 0xbe, 0xbd, 0xcc,
            0xbb, 0xdc, 0xbe, 0xba, 0xbd, 0xcd, 0xbb, 0xab, 0xdc, 0xbc,
            0xbd, 0xdd, 0xbc, 0xab, 0xcd, 0xbb, 0xcc, 0xdd, 0xfd, 0xdb,
            0xbc, 0xbc, 0xba, 0xca, 0xbd, 0xcc, 0xde, 0xba, 0xbb, 0xbd,
            0xec, 0xab, 0xba, 0xed, 0xcd, 0xbb, 0xde, 0xbb, 0xbb, 0xab,
            0xca, 0xcd, 0xbb, 0xcb, 0xdb, 0xcc, 0xcd, 0xdd, 0xbc, 0xda,
            0xed, 0xbb, 0xba, 0xcd, 0xeb, 0xce, 0xdd, 0xeb, 0xad, 0x9d,
            0xba, 0xda, 0xbb, 0xc9, 0x5b, 0x9c, 0xeb, 0xbb, 0xcc, 0xca,
            0xdb, 0xc9, 0x9b, 0x8d, 0xdc, 0xbb, 0xad, 0xab, 0xdc, 0xdc,
            0xcc, 0xad, 0xba, 0x4c, 0xdd, 0xbb, 0xbb, 0x8e, 0xcc, 0xbd,
            0xcc, 0xbc, 0xcd, 0xeb, 0xbe, 0xac, 0xbd, 0xdd, 0xcc, 0xbd,
            0xcc, 0xcb, 0xdc, 0xcc, 0xdc, 0xbc, 0xcc, 0xcb, 0xcb, 0xcc,
            0xbc, 0xcb, 0xcc, 0xac, 0xcd, 0xbc, 0xcd, 0xcc, 0xcb, 0xcc,
            0xab, 0xbc, 0xce, 0xdc, 0xb9, 0xac, 0xed, 0xcc, 0xcd, 0xbc,
            0xce, 0xbb, 0xfd, 0xcc, 0xbc, 0xcd, 0xba, 0xbb, 0x9c, 0xba,
            0xac, 0xad, 0xba, 0xca, 0xec, 0xac, 0x9b, 0xdb, 0xcb, 0xba,
            0xdb, 0xbe, 0xbd, 0xbc, 0x99, 0xcb, 0xbc, 0xcb, 0xac, 0xbb,
            0x9b, 0xfa, 0xdb, 0xcd, 0xbb, 0xbc, 0xbb, 0xbb, 0xee, 0xbb,
            0xdc, 0xbc, 0xcc, 0xec, 0xbb, 0xbd, 0xbb, 0xdc, 0xcc, 0x9b,
            0xca, 0xbb, 0xbb, 0xaf, 0xcc, 0xca, 0xdd, 0xbc, 0xba, 0xcb,
            0xdc, 0xdc, 0xea, 0xab, 0xba, 0xca, 0xcb, 0xcc, 0xbd, 0xac,
            0xcc, 0xcb, 0xcb, 0xba, 0xba, 0xab, 0xcd, 0xdd, 0xac, 0xec,
            0xed, 0xdb, 0xdb, 0xac, 0xcc, 0xce, 0xbb, 0xbd, 0xdc, 0xbb,
            0xca, 0xbc, 0xcb, 0xcc, 0xcb, 0xce, 0xcb, 0xcb, 0xdd, 0xcb,
            0xcc, 0xbc, 0xec, 0xbd, 0xeb, 0xdb, 0xcd, 0xcd, 0xce, 0xbd,
            0xac, 0xdc, 0xec, 0xdb, 0xcc, 0xcc, 0xcd, 0xcc, 0xbc, 0xbb,
            0xed, 0xbb, 0xcd, 0xbd, 0xcd, 0xcc, 0xca, 0xcd, 0xbd, 0xcb,
            0xba, 0xbb, 0xdb, 0xcb, 0xbd, 0xcc, 0xba, 0xec, 0xbc, 0xac,
            0xcd, 0xbb, 0xac, 0xea, 0xcc, 0xdc, 0xbc, 0xcd, 0xbb, 0xbd,
            0xcd, 0xbb, 0xec, 0xcb, 0xbb, 0xbb, 0xab, 0xbb, 0xdd, 0xaf,
            0xcc, 0xdb, 0xdb, 0xbb, 0xbb, 0xbc, 0xed, 0xbb, 0xec, 0xde,
            0xbb, 0xcc, 0xbd, 0xaf, 0x9d, 0xfc, 0xfc, 0xbb, 0xbb, 0xcb,
            0xbd, 0xcc, 0x3e, 0xef, 0x76, 0x3e, 0x5d, 0x6e, 0xa9, 0xbe,
            0xf6, 0x1a, 0xae, 0xbf, 0x52, 0xbc, 0x3d, 0xbf, 0x58, 0x40,
            0x81, 0x3d, 0xa8, 0x54, 0xe2, 0x3d, 0x95, 0xbc, 0x02, 0x3f,
            0x97, 0x14, 0x1a, 0x3f, 0xf1, 0x34, 0xdf, 0xbe, 0xf5, 0xf0,
            0xdf, 0xbe, 0x8a, 0x12, 0xbd, 0x3e, 0x80, 0x8a, 0x10, 0x3f,
            0xe0, 0x17, 0xb8, 0xbc, 0xbd, 0xdd, 0x31, 0x3f, 0x17, 0x9f,
            0x18, 0xbf, 0x41, 0x95, 0xc3, 0x3e, 0x8e, 0xae, 0x52, 0x3f,
            0x4c, 0x46, 0x4b, 0x3f, 0x24, 0x9c, 0xe4, 0xbd, 0x72, 0x9c,
            0x8f, 0x3d, 0x23, 0x2e, 0x9f, 0xbc, 0x41, 0x25, 0x13, 0x3e,
            0x46, 0xaa, 0xfb, 0x3d, 0x48, 0xb9, 0x53, 0xbe, 0xc7, 0xbd,
            0x66, 0x3f, 0xbc, 0xc7, 0x44, 0x3e, 0x88, 0x45, 0xd4, 0x3c,
            0xd6, 0x6e, 0x3f, 0xbf, 0x6f, 0x32, 0x3c, 0x3f, 0xfa, 0x20,
            0x82, 0x3f, 0x78, 0x6a, 0xb5, 0x3f, 0x9a, 0x2e, 0xbf, 0x3e,
            0xf8, 0x8d, 0x8f, 0x3e, 0x88, 0x45, 0x9e, 0xbf, 0x34, 0x80,
            0xef, 0x3e, 0x1f, 0x17, 0xac, 0x3e, 0x87, 0xe0, 0x4e, 0x3f,
            0x37, 0x68, 0x56, 0x3f, 0x42, 0x17, 0x52, 0x3e, 0x3a, 0xe1,
            0x24, 0xbf, 0x5e, 0x5b, 0xce, 0x3e, 0xec, 0xce, 0x90, 0x3d,
            0x57, 0xd3, 0xc2, 0x3e, 0xf5, 0x73, 0x4a, 0x3f, 0x55, 0x11,
            0x06, 0xbd, 0xe2, 0xd9, 0xf4, 0xbe, 0xd7, 0xb6, 0x87, 0xbf,
            0x83, 0x0a, 0xb1, 0x3e, 0xa3, 0xd1, 0x5c, 0x3f, 0xbd, 0xc9,
            0x1f, 0x40, 0x39, 0xb0, 0x15, 0xbc, 0x0b, 0xaf, 0x1b, 0x3f,
            0x3c, 0x63, 0x36, 0x3f, 0x6d, 0xa5, 0xc7, 0x3e, 0x44, 0x97,
            0x1d, 0xbf, 0x38, 0xf9, 0x6b, 0x3e, 0x80, 0x3b, 0x7b, 0xbf,
            0x6c, 0xf6, 0xf2, 0x3e, 0xb2, 0xff, 0xa2, 0x3f, 0x9f, 0x75,
            0x54, 0x3f, 0x7f, 0x39, 0xb4, 0x3d, 0x64, 0x93, 0x45, 0xbe,
            0x45, 0x90, 0xf6, 0x3e, 0xb9, 0x53, 0xae, 0x3e, 0xd8, 0x84,
            0xf2, 0xbf, 0x42, 0xd0, 0xc4, 0xbf, 0x38, 0xd1, 0x82, 0xbf,
            0x02, 0x40, 0x45, 0xbf, 0x5b, 0xed, 0x03, 0xbf, 0xd0, 0x86,
            0x5f, 0xbe, 0x1c, 0x02, 0x0e, 0x3c, 0x6d, 0x99, 0x80, 0x3e,
            0x9c, 0x89, 0x0b, 0x3f, 0x42, 0x38, 0x5f, 0x3f, 0xd4, 0x6e,
            0xab, 0x3f, 0x97, 0x00, 0xd3, 0x3f, 0x2a, 0x22, 0xf2, 0x3f,
            0x42, 0x9c, 0x0a, 0x40, 0x9b, 0xc0, 0x29, 0x40, 0xf7, 0x89,
            0x3e, 0x40, 0x77, 0xb7, 0x84, 0x64, 0x99, 0x37, 0x77, 0x57,
            0x78, 0x82, 0x88, 0x77, 0x36, 0x82, 0x48, 0xd6, 0x67, 0x77,
            0x66, 0xa8, 0x53, 0x56, 0x88, 0x7f, 0x84, 0x83, 0x07, 0x88,
            0xd5, 0x24, 0x77, 0x76, 0x66, 0x66, 0x66, 0x65, 0x56, 0x67,
            0x56, 0x76, 0x67, 0x66, 0x56, 0x65, 0x65, 0x66, 0x57, 0x66,
            0x77, 0x55, 0x55, 0x56, 0x55, 0x65, 0x66, 0x75, 0x57, 0x67,
            0x66, 0x57, 0x67, 0x67, 0x66, 0x76, 0x66, 0xa8, 0x84, 0x64,
            0x99, 0x45, 0x78, 0x56, 0x58, 0x93, 0x8a, 0x88, 0x38, 0x84,
            0x55, 0xb5, 0x88, 0x85, 0x45, 0x99, 0x44, 0x46, 0x78, 0x6d,
            0x55, 0x93, 0x16, 0x88, 0xa6, 0x33, 0x76, 0x65, 0x74, 0xa9,
            0x84, 0x36, 0x78, 0x25, 0x79, 0x44, 0x99, 0x64, 0x68, 0x76,
            0x27, 0x84, 0x37, 0xc7, 0x75, 0x87, 0x87, 0x97, 0x42, 0x57,
            0x77, 0x4c, 0x23, 0x73, 0x23, 0x69, 0xa6, 0x53, 0x97, 0x43,
            0x77, 0x48, 0x65, 0x86, 0x78, 0x63, 0x65, 0x77, 0x66, 0x57,
            0x68, 0x56, 0x56, 0x66, 0x54, 0x76, 0x87, 0x65, 0x56, 0x68,
            0x95, 0x55, 0x56, 0x6a, 0x6a, 0x86, 0x44, 0x68, 0x44, 0x35,
            0x47, 0x43, 0x47, 0x95, 0x53, 0x67, 0x77, 0x74, 0x76, 0x45,
            0x68, 0x85, 0x34, 0x55, 0x66, 0x76, 0x77, 0xa7, 0x53, 0x56,
            0x77, 0x77, 0x57, 0x35, 0x73, 0x48, 0x77, 0x66, 0x56, 0x84,
            0x65, 0x74, 0x66, 0x98, 0x66, 0x86, 0x56, 0x76, 0x78, 0x66,
            0x57, 0x64, 0x69, 0x86, 0x46, 0x64, 0x55, 0x65, 0x64, 0xd5,
            0x68, 0x58, 0x84, 0x75, 0x66, 0x88, 0x76, 0x5a, 0x24, 0x65,
            0x35, 0x69, 0x95, 0x62, 0x58, 0x54, 0x68, 0x17, 0x36, 0x88,
            0x46, 0x74, 0x43, 0x88, 0x84, 0x37, 0x53, 0x45, 0x75, 0x57,
            0x57, 0x46, 0x75, 0x47, 0x77, 0x46, 0xb6, 0x65, 0x45, 0x68,
            0x8a, 0x68, 0x75, 0x47, 0x24, 0x47, 0x28, 0x53, 0xbb, 0xdb,
            0x3e, 0x3d, 0xe5, 0x36, 0x01, 0xbd, 0x53, 0xf2, 0x88, 0xbf,
            0x5b, 0xe1, 0xad, 0x3d, 0x9f, 0x70, 0xe2, 0xbf, 0x5b, 0x49,
            0x5c, 0x3e, 0xc9, 0x22, 0x6c, 0x3d, 0x1c, 0x56, 0xe9, 0x3e,
            0xd8, 0x59, 0x42, 0xbe, 0xa2, 0x0f, 0x85, 0xbe, 0xae, 0xde,
            0xdf, 0x3e, 0x16, 0x66, 0x0a, 0xbc, 0x9b, 0xbf, 0xe9, 0xbe,
            0x35, 0xf7, 0x82, 0xbc, 0xa4, 0x62, 0xcc, 0xbc, 0x04, 0x9a,
            0xb6, 0x3e, 0x1f, 0xd5, 0x95, 0xbb, 0x5a, 0xfb, 0x8f, 0xbe,
            0x58, 0x74, 0xd1, 0x3c, 0x39, 0xd9, 0xa9, 0x3c, 0xd8, 0x65,
            0xd5, 0xbc, 0x57, 0xee, 0x14, 0xbd, 0xfa, 0x86, 0x40, 0xbd,
            0x3e, 0xcf, 0xb5, 0x3c, 0x1c, 0x9f, 0x01, 0xbf, 0x15, 0x15,
            0xfc, 0x3d, 0x5d, 0xad, 0x83, 0x3f, 0x1e, 0x4a, 0x0a, 0x40,
            0x48, 0xff, 0xf5, 0x3f, 0xc3, 0x56, 0x95, 0xc0, 0x16, 0xf3,
            0xde, 0xc0, 0xe2, 0x90, 0xcb, 0x3e, 0x33, 0xf6, 0x3d,
            0x3f, 0x3c, 0x1a, 0xf9, 0xbe, 0xb2, 0x61, 0x0a, 0x40,
            0x52, 0x77, 0x32, 0x41, 0x2f, 0xbf, 0x1d, 0x3f, 0xa2,
            0x83, 0x2c, 0xc1, 0xaa, 0x04, 0x9b, 0xc1, 0xa5, 0xe3,
            0x38, 0xc0, 0x97, 0x4d, 0x4c, 0x3f, 0x16, 0xc5, 0x65,
            0x3e, 0xc6, 0x68, 0x81, 0xc0, 0x0d, 0x32, 0x28, 0xbf
    };

    return AI_HANDLE_PTR(s_relu_2_8_weights);

}
