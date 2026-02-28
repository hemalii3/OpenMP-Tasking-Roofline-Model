#include <unordered_map>
#include <omp.h>
#include <algorithm> 
#include <limits>    
#include <cmath>     
#include "helpers.hpp"

unsigned long SequenceInfo::gpsa_sequential(float** S) {
    unsigned long visited = 0;

    for (unsigned int i = 1; i < rows; i++) {
        S[i][0] = i * gap_penalty;
        visited++;
    }

    for (unsigned int j = 0; j < cols; j++) {
        S[0][j] = j * gap_penalty;
        visited++;
    }
    
    for (unsigned int i = 1; i < rows; i++) {
        for (unsigned int j = 1; j < cols; j++) {
            float match  = S[i - 1][j - 1] + (X[i - 1] == Y[j - 1] ? match_score : mismatch_score);
            float del    = S[i - 1][j] + gap_penalty;
            float insert = S[i][j - 1] + gap_penalty;
            
            S[i][j] = std::max(match, std::max(del, insert));
            visited++;
        }
    }

    return visited;
}

unsigned long SequenceInfo::gpsa_taskloop(float** S, long grain_size=1, int block_size_x=1, int block_size_y=1) {
    for (unsigned int i = 1; i < rows; i++) {
        S[i][0] = i * gap_penalty;
    }
    for (unsigned int j = 0; j < cols; j++) {
        S[0][j] = j * gap_penalty;
    }

    int Rx = (rows - 1 + block_size_x - 1) / block_size_x; 
    int Ry = (cols - 1 + block_size_y - 1) / block_size_y; 
    int total_blocks_k = Rx + Ry - 1; 

    #pragma omp parallel
    {
        #pragma omp single nowait
        {
            for (int k = 1; k <= total_blocks_k; ++k) {
                
                int i_start = std::max(1, k - Ry + 1);
                int i_end = std::min(Rx, k);
                
                #pragma omp taskloop \
                    shared(S, X, Y, rows, cols, match_score, mismatch_score, gap_penalty, block_size_x, block_size_y)
                for (int ix = i_start; ix <= i_end; ++ix) {
                    int jy = k - ix + 1; 
                    
                    unsigned int i_block_start = (ix - 1) * block_size_x + 1;
                    unsigned int j_block_start = (jy - 1) * block_size_y + 1;
                    
                    unsigned int i_block_end = std::min((unsigned int)rows - 1, i_block_start + block_size_x - 1);
                    unsigned int j_block_end = std::min((unsigned int)cols - 1, j_block_start + block_size_y - 1);

                    for (unsigned int i = i_block_start; i <= i_block_end; ++i) {
                        for (unsigned int j = j_block_start; j <= j_block_end; ++j) {
                            float match = S[i - 1][j - 1] + (X[i - 1] == Y[j - 1] ? match_score : mismatch_score);
                            float del = S[i - 1][j] + gap_penalty;
                            float insert = S[i][j - 1] + gap_penalty;
                            S[i][j] = std::max(match, std::max(del, insert));
                        }
                    }
                } 
            }
        } 
    } 
    
    return (unsigned long)(rows - 1) * (cols - 1) + (rows - 1) + cols; 
}


unsigned long SequenceInfo::gpsa_tasks(float** S, long grain_size, int block_size_x, int block_size_y) {
    for (unsigned int i = 1; i < rows; i++) {
        S[i][0] = i * gap_penalty;
    }
    for (unsigned int j = 0; j < cols; j++) {
        S[0][j] = j * gap_penalty;
    }

    int Rx = (rows - 1 + block_size_x - 1) / block_size_x;
    int Ry = (cols - 1 + block_size_y - 1) / block_size_y;
    int total_blocks_k = Rx + Ry - 1;

    #pragma omp parallel
    #pragma omp single
    {
        for (int k = 1; k <= total_blocks_k; ++k) {
            int i_start = std::max(1, k - Ry + 1);
            int i_end = std::min(Rx, k);

            for (int ix = i_start; ix <= i_end; ++ix) {
                int jy = k - ix + 1;

                unsigned int i_block_start = (ix - 1) * block_size_x + 1;
                unsigned int j_block_start = (jy - 1) * block_size_y + 1;
                unsigned int i_block_end = std::min((unsigned int)rows - 1, i_block_start + block_size_x - 1);
                unsigned int j_block_end = std::min((unsigned int)cols - 1, j_block_start + block_size_y - 1);

                #pragma omp task \
                    shared(S, X, Y, match_score, mismatch_score, gap_penalty) \
                    firstprivate(i_block_start, j_block_start, i_block_end, j_block_end)
                {
                    for (unsigned int i = i_block_start; i <= i_block_end; ++i) {
                        for (unsigned int j = j_block_start; j <= j_block_end; ++j) {
                            float match = S[i - 1][j - 1] + (X[i - 1] == Y[j - 1] ? match_score : mismatch_score);
                            float del = S[i - 1][j] + gap_penalty;
                            float insert = S[i][j - 1] + gap_penalty;
                            S[i][j] = std::max(match, std::max(del, insert));
                        }
                    }
                }
            } 

            #pragma omp taskwait
        } 
    } 

    return (unsigned long)(rows - 1) * (cols - 1) + (rows - 1) + cols;
}
