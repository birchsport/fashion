package com.shopstyle.dl.fashion.configuration;

import java.io.File;
import java.io.IOException;
import java.util.Optional;
import java.util.Random;

import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.BaseImageLoader;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.dataset.api.preprocessor.VGG16ImagePreProcessor;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;
import org.slf4j.Logger;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.util.StopWatch;

@Configuration
public class DL4JConfiguration {

	private static final Logger log = org.slf4j.LoggerFactory.getLogger(DL4JConfiguration.class);
	private static final String[] allowedExtensions = BaseImageLoader.ALLOWED_FORMATS;
	private static final String COMPUTATION_GRAPH_FILE_NAME = "dogscats_comp_graph.dl4j";

	private static final long seed = 12345;

	private static final Random randNumGen = new Random(seed);

	private static final int height = 224;
	private static final int width = 224;
	private static final int channels = 3;

	@Bean
	public Model vgg16() {
		int rngSeed = 123; // random number seed for reproducibility
		double rate = 0.0015; // learning rate
		int numClasses = 2;

		Optional<ComputationGraph> optional = loadComputationalGraph(new File(COMPUTATION_GRAPH_FILE_NAME));
		if (optional.isPresent()) {
			log.info("Loaded pretrained model:");
			ComputationGraph computationGraph = optional.get();
			log.info(computationGraph.summary());
			validateSingleImage(computationGraph);
			return computationGraph;
		}

		log.info("Building model....");
		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().seed(rngSeed)
				.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).iterations(1)
				.activation(Activation.RELU).weightInit(WeightInit.XAVIER).learningRate(rate)
				.updater(new Nesterovs(0.98)).regularization(true).l2(rate * 0.005).list()
				.layer(0, new DenseLayer.Builder().nIn(width * height * channels).nOut(500).build())
				.layer(1, new DenseLayer.Builder().nIn(500).nOut(100).build())
				.layer(2,
						new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD).activation(Activation.SOFTMAX)
								.nIn(100).nOut(numClasses).build())
				.pretrain(false).backprop(true).setInputType(InputType.convolutional(height, width, channels)).build();

		MultiLayerNetwork network = new MultiLayerNetwork(conf);
		network.init();
		network.setListeners(new ScoreIterationListener(5));

		log.info("Network summary:");
		log.info(network.summary());
		StopWatch stopWatch = new StopWatch();
		stopWatch.start();
		log.info("Fitting....");
		network.fit(loadTrainData());
		log.info("Fit complete. ELapsed time: [{}]", stopWatch.toString());
		evaluateNetwork(network);
		saveNetwork(network, new File(COMPUTATION_GRAPH_FILE_NAME));
		return network;

	}

	private void validateSingleImage(ComputationGraph graph) {
		File dir = new File(System.getProperty("user.home"), "/data/dogscats/train/dogs");
		File file = new File(dir, "dog.9993.jpg");
		log.info("Validating: {}", file.getAbsolutePath());
		NativeImageLoader loader = new NativeImageLoader(height, width, channels);
		try {
			INDArray image = loader.asMatrix(file);
			DataNormalization scaler = new VGG16ImagePreProcessor();
			scaler.transform(image);
			INDArray[] output = graph.output(false, image);
			log.info("Predictions: {}", file.getAbsolutePath(), output[0]);
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	private static DataSetIterator loadTrainData() {
		FileSplit split = new FileSplit(new File(System.getProperty("user.home"), "/data/dogscats/train"),
				NativeImageLoader.ALLOWED_FORMATS, randNumGen);
		ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
		// ImageTransform flipTransform1 = new FlipImageTransform(randNumGen);
		// ImageTransform flipTransform2 = new FlipImageTransform(new Random(123));
		// ImageTransform warpTransform = new WarpImageTransform(randNumGen, 42);
		// List<ImageTransform> transforms = Arrays
		// .asList(new ImageTransform[] { flipTransform1, warpTransform, flipTransform2
		// });

		ImageRecordReader recordReader = new ImageRecordReader(height, width, channels, labelMaker);

		try {
			recordReader.initialize(split);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		RecordReaderDataSetIterator dataIter = new RecordReaderDataSetIterator(recordReader, 4, 1, 2);
		DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
		scaler.fit(dataIter);
		dataIter.setPreProcessor(scaler);

		return dataIter;

	}

	private static DataSetIterator loadTestData() {
		FileSplit split = new FileSplit(new File(System.getProperty("user.home"), "/data/dogscats/valid"),
				NativeImageLoader.ALLOWED_FORMATS, randNumGen);
		ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
		// ImageTransform flipTransform1 = new FlipImageTransform(randNumGen);
		// ImageTransform flipTransform2 = new FlipImageTransform(new Random(123));
		// ImageTransform warpTransform = new WarpImageTransform(randNumGen, 42);
		// List<ImageTransform> transforms = Arrays
		// .asList(new ImageTransform[] { flipTransform1, warpTransform, flipTransform2
		// });

		ImageRecordReader recordReader = new ImageRecordReader(height, width, channels, labelMaker);

		try {
			recordReader.initialize(split);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		RecordReaderDataSetIterator dataIter = new RecordReaderDataSetIterator(recordReader, 4, 1, 2);
		DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
		scaler.fit(dataIter);
		dataIter.setPreProcessor(scaler);

		return dataIter;

	}

	private Optional<ComputationGraph> loadComputationalGraph(File file) {
		ComputationGraph pretrainedNet = null;
		if (file.exists()) {
			try {
				pretrainedNet = ModelSerializer.restoreComputationGraph(file);
			} catch (IOException e1) {
				e1.printStackTrace();
			}
		}
		return Optional.ofNullable(pretrainedNet);
	}

	private void saveNetwork(MultiLayerNetwork network, File file) {
		try {
			ModelSerializer.writeModel(network, file, false);
		} catch (IOException e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		}
	}

	private void evaluateNetwork(MultiLayerNetwork network) {
		log.info("Evaluate network....");
		Evaluation eval = new Evaluation(2); // create an evaluation object with 10 possible classes
		DataSetIterator testData = loadTestData();
		while (testData.hasNext()) {
			DataSet next = testData.next();
			INDArray output = network.output(next.getFeatureMatrix()); // get the networks prediction
			eval.eval(next.getLabels(), output); // check the prediction against the true class
		}

		log.info(eval.stats());
	}
}
