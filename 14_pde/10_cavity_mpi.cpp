#include <vector>
#include <cstdio>
#include <cmath>
#include <chrono>
using namespace std;
#include <mpi.h>

int main(int argc, char **argv){
    const int nx = 4100;
    const int ny = 4100;
    const int nt = 10;
    const int nit = 50;
    const float dx = 2.0 / (nx - 1);
    const float dy = 2.0 / (ny - 1);
    const float dt = 0.001;
    const int rho = 1;
    const float nu = 0.02;
    float u[ny*nx],v[ny*nx],p[ny][nx],b[ny][nx],pn[ny][nx],un[ny][nx],vn[ny][nx];

    MPI_Init(&argc, &argv);
    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int begin = rank * ((ny-1) / size);
    int end = (rank + 1) * ((ny-1) / size);
    //printf("rank: %d/%d begin:%d end:%d \n",rank,size,begin,end);

    for(int n=0; n<nt; n++){
        auto tic = chrono::steady_clock::now();

        for(int j=begin; j<end; j++){
            if ((rank==0 && j==begin)){
                    continue;
            }
            for(int i=1; i<nx-1; i++){
                b[j][i] = rho * (1 / dt *
                    ((u[j*nx+i+1] - u[j*nx+i-1]) / (2 * dx) + (v[(j+1)*nx+i] - v[(j-1)*nx+i]) / (2 * dy)) -
                    pow((u[j*nx+i+1] - u[j*nx+i-1]) / (2 * dx),2) - 2 * ((u[(j+1)*nx+i] - u[(j-1)*nx+i]) / (2 * dy) *
                     (v[j*nx+i+1] - v[j*nx+i-1]) / (2 * dx)) - pow((v[(j+1)*nx+i] - v[(j-1)*nx+i]) / (2 * dy),2));
            }
        }
        MPI_Allgather(&b[begin*nx], (nx)*(end-begin), MPI_FLOAT, &b, (nx)*(end-begin), MPI_FLOAT, MPI_COMM_WORLD);

        for(int it=0; it<nit; it++){
            // pn = p.copy()
            for(int j=0; j<ny; j++){
                for(int i=0; i<nx; i++){
                    pn[j][i] = p[j][i];
                }
            }
            for(int j=begin; j<end; j++){
                if ((rank==0 && j==begin)){
                        continue;
                }
                for(int i=1; i<nx-1; i++){
                    p[j][i] = ( pow(dy, 2) * (pn[j][i+1] + pn[j][i-1]) +
                                pow(dx, 2) * (pn[j+1][i] + pn[j-1][i]) -
                                b[j][i] * pow(dx, 2) * pow(dy, 2)) / (2 * (pow(dx, 2) + pow(dy,2)));
                }
            }
            MPI_Allgather(&p[begin*nx], (nx)*(end-begin), MPI_FLOAT, &p, (nx)*(end-begin), MPI_FLOAT, MPI_COMM_WORLD);
            
            for(int j=0; j<ny; j++) p[j][nx-1] = p[j][nx-2];
            for(int i=0; i<nx; i++) p[0][i] = p[1][i];
            for(int j=0; j<ny; j++) p[j][0] = p[j][1];
            for(int i=0; i<nx; i++) p[ny-1][i] = 0;
        }

        for(int j=0; j<ny; j++){
            for(int i=0; i<nx; i++){
                un[j][i] = u[j*nx+i];
                vn[j][i] = v[j*nx+i];
            }
        }
        
        for(int j=begin; j<end; j++){
            if ((rank==0 && j==begin)){
                    continue;
            }
            for(int i=1; i<nx-1; i++){
                
                u[j*nx+i] = un[j][i] - un[j][i] * dt / dx * (un[j][i] - un[j][i - 1])
                                   - un[j][i] * dt / dy * (un[j][i] - un[j - 1][i])
                                   - dt / (2 * rho * dx) * (p[j][i+1] - p[j][i-1])
                                   + nu * dt / pow(dx, 2) * (un[j][i+1] - 2 * un[j][i] + un[j][i-1])
                                   + nu * dt / pow(dy, 2) * (un[j+1][i] - 2 * un[j][i] + un[j-1][i]);
                v[j*nx+i] = vn[j][i] - vn[j][i] * dt / dx * (vn[j][i] - vn[j][i - 1])
                                   - vn[j][i] * dt / dy * (vn[j][i] - vn[j - 1][i])
                                   - dt / (2 * rho * dx) * (p[j+1][i] - p[j-1][i])
                                   + nu * dt / pow(dx, 2) * (vn[j][i+1] - 2 * vn[j][i] + vn[j][i-1])
                                   + nu * dt / pow(dy, 2) * (vn[j+1][i] - 2 * vn[j][i] + vn[j-1][i]);
            }
        }
        MPI_Allgather(&u[begin*nx], (nx)*(end-begin), MPI_FLOAT, &u, (nx)*(end-begin), MPI_FLOAT, MPI_COMM_WORLD);
        MPI_Allgather(&v[begin*nx], (nx)*(end-begin), MPI_FLOAT, &v, (nx)*(end-begin), MPI_FLOAT, MPI_COMM_WORLD);
        
        for(int i=0; i<nx; i++){
            u[i] = 0;
            u[(ny-1)*nx+i] = 1;
            v[i] = 0;
            v[(ny-1)*nx+i] = 0;
        }
        for(int j=0; j<ny; j++){
            u[j*nx+0] = 0;
            u[j*nx+nx-1] = 0;
            v[j*nx+0] = 0;
            v[j*nx+nx-1] = 0;
        }
        
        if (rank==0){
            auto toc = chrono::steady_clock::now();
            float time = chrono::duration<float>(toc-tic).count();
            printf("step=%d: %lf [s]\n",n,time);
        }
        
    }

    MPI_Finalize();
}