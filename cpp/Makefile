# Makefile for compiling C++ project

# Compiler
CXX = g++

# Compiler options
CXXFLAGS = -std=c++11 -Wall -Wextra

# Linker options
LDFLAGS =

# Include path
INC_PATH = -I./include -ID:/Users/Jerome/Documents/Ingenierie/Programmation/eigen-3.4.0

# Source path
SRC_PATH = ./src

# Build path
BUILD_PATH = ./build

# Source files
SRCS := $(wildcard $(SRC_PATH)/*.cpp)

# Object files
OBJS := $(patsubst $(SRC_PATH)/%.cpp,$(BUILD_PATH)/%.o,$(SRCS))

# Executable
EXEC = $(BUILD_PATH)/CubicLagrangeMinimize

# Targets
all: $(EXEC)

$(EXEC): $(OBJS)
	$(CXX) $(LDFLAGS) $(OBJS) -o $(EXEC)

$(BUILD_PATH)/%.o: $(SRC_PATH)/%.cpp
	$(CXX) $(CXXFLAGS) $(INC_PATH) -c $< -o $@

clean:
	rm -f $(OBJS) $(EXEC)
