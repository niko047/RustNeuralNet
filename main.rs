extern crate core;


fn main() {
    let mut nn = NeuralNet { weights : vec![0.32, 0.65], learning_rate : 0.00001};
    let mut xs = vec![];
    for i in -2000..2000 {
        xs.push(i as f32 * 0.001)
    }
    let samples = nn.sample_datapoints(xs);
    for i in 1..5000 {
        nn.forward_pass_and_loss(&samples)
    };
    println!("{:?}", nn.weights);
}

struct NeuralNet {
    weights : Vec<f32>,
    learning_rate : f32
}

fn dotprod(a: &Vec<f32>, b: &Vec<f32>) -> f32 {
    /// a : vector of size Nx1
    /// b : vector of size Nx1
    /// :return:
    /// Scalar given by the dot product of a and b

    assert_eq!(a.len(), b.len());
    let mut sum = 0 as f32;
    for (i, elem) in a.iter().enumerate() {
        sum += elem * b[i];
    };
    sum
}

fn vec_times_scalar(a: &Vec<f32>, k: &f32) -> Vec<f32> {
    /// a : vector of size Nx1
    /// k : scalar
    /// :return:
    /// Vector given by the multiplication of a and k
    ///
    let mut resvec = Vec::new();
    for i in a {
        resvec.push(i*k)
    };
    resvec
}

enum Operator {
    minus,
    plus,
    multiply
}

fn vec_2_vec(a: &Vec<f32>, b: &Vec<f32>, o: Operator) -> Vec<f32> {
    /// a : vector of size Nx1
    /// b : vector of size Nx1
    /// :return:
    /// Scalar given by the sum of a and b

    assert_eq!(a.len(), b.len());
    let mut resvec = Vec::new();
    for (i, elem) in a.iter().enumerate() {
        match o {
            Operator::minus => resvec.push(elem - b[i]),
            Operator::plus => resvec.push(elem + b[i]),
            Operator::multiply => resvec.push(elem*b[i])
        }
    };
    resvec
}

fn vec_pow(a: &Vec<f32>, n: &i32) -> Vec<f32> {
    let mut resvec = Vec::new();
    for i in a {
        resvec.push(f32::powi(*i, *n))
    };
    resvec
}

fn vec_plus_scalar(a: &Vec<f32>, k: &f32) -> Vec<f32> {
    /// a : vector of size Nx1
    /// k : scalar
    /// :return:
    /// Vector given by the addition of a and k
    ///
    let mut resvec = Vec::new();
    for i in a {
        resvec.push(i+k)
    };
    resvec
}

fn vec_sqrt(a: &Vec<f32>) -> Vec<f32> {
    let mut resvec = Vec::new();
    for i in a {
      resvec.push(i.sqrt())
    };
    resvec
}

fn sum(a: &Vec<f32>) -> f32 {
    let mut sum = 0 as f32 ;
    for i in a {
      sum += *i
    };
    sum
}

impl NeuralNet {
    fn compute_true_function(&self, value: &f32) -> f32 {
        // True function from which we're going to be drawing data samples
        f32::powi(*value, 4) - 2.0 * f32::powi(*value, 2)
    }

    fn sample_datapoints(&self, arr: Vec<f32>) -> Vec<Vec<f32>> {
        let mut output : Vec<Vec<f32>> = Vec::new();
        let mut v : Vec<f32> = Vec::new();
        for i in &arr {
            v.push(self.compute_true_function(i));
        };
        output.push(arr);
        output.push(v);
        output
    }
    fn forward_pass_and_loss(&mut self, data: &Vec<Vec<f32>>) {
        // 1. Compute the estimated function, we assume it to be quadratic
        let mut yhat = vec_plus_scalar(
            &vec_times_scalar(&vec_pow(&data[0], &2), &self.weights[1]),
            &self.weights[0]
        );

        // 2. Compute the loss
        let diff = vec_2_vec(&yhat, &data[1], Operator::minus);
        let loss = sum(&vec_pow(&diff, &2));
        println!("{}", loss);
        let dJ_dy = vec_times_scalar(&diff, &2.0);

        let dy_da : f32 = sum(&dJ_dy);
        let dy_dc: f32 = sum(&vec_2_vec(&dJ_dy, &vec_pow(&data[0], &2), Operator::multiply));

        self.weights[0] -= self.learning_rate * dy_da;
        self.weights[1] -= self.learning_rate * dy_dc;
    }
}