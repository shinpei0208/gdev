
all:
	gcc -L /usr/local/gdev/lib64 -I /usr/local/gdev/include --shared -fPIC -o libsrv_test.so srv_test.c
	$(if $(BUILD_DIR), cp libsrv_test.so test.cubin $(BUILD_DIR)/ )

clean:
	rm -f *.o *.so
