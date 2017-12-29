OPCV = -L/usr/local/lib -lopencv_core -lopencv_highgui -lopencv_features2d -lopencv_flann -lopencv_imgproc -lopencv_imgcodecs
NCS = -lmvnc

.PHONY: all
all: compile


.PHONY: compile
compile: build_dir
	@echo "Compiling test.cpp"
	g++ src/test.cpp src/fp16.c -o build/test $(NCS) $(OPCV)
	@echo "Finished building test"

.PHONY: run
run: compile
	@echo "\nStarting application"
	./build/test

build_dir:
	@if test ! -d build; \
	then mkdir build; \
	fi

clean:
	@echo "Cleaning programs";
	rm -rf build
