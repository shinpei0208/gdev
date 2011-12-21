
all: srv_gaes.c
	gcc -L /usr/local/gdev/lib64 -I /usr/local/gdev/include -O2 --shared -fPIC -o libsrv_gaes.so srv_gaes.c
	nvcc -arch sm_20 -cubin gaes.cu -o gaes.cubin
	$(if $(BUILD_DIR), cp libsrv_gaes.so gaes.cubin $(BUILD_DIR)/ ) 

clean:
	rm -f *.o *.so *.cubin
