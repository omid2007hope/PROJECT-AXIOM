#ifndef include_training
#define include_training

struct Train {

  bool training_in_use = false;

}

class Training {

public:
  // Imaginary function of training to develop optimizer for now
  // basically fake right now just to call

  void training(const Train &train);
};

#endif
