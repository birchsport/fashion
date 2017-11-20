package com.shopstyle.dl.fashion.configuration;

import java.io.File;
import java.io.IOException;
import java.util.Optional;
import java.util.Random;

import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
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
	private static final String COMPUTATION_GRAPH_FILE_NAME = "fashion_comp_graph.zip";

	private static final long seed = 12345;

	private static final Random randNumGen = new Random(seed);

//	private static final int height = 164;
//	private static final int width = 205;
	private static final int height = 224;
	private static final int width = 224;
	private static final int channels = 3;
	private static final int batchSize = 64;
	private static final int numEpochs = 1;
	private int numClasses = 1;
	private RecordReaderDataSetIterator trainDataIter;
	private RecordReaderDataSetIterator testDataIter;

	@Bean
	public MultiLayerNetwork createNetwork() {
		double rate = 0.006; // learning rate
		loadData();
		Optional<MultiLayerNetwork> optional = loadComputationalGraph(new File(COMPUTATION_GRAPH_FILE_NAME));
		if (optional.isPresent()) {
			log.info("Loaded pretrained model:");
			MultiLayerNetwork network = optional.get();
			log.info(network.summary());
			// validateSingleImage(network);
			evaluateNetwork(network);
			return network;
		}

		log.info("Building model....");
		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().seed(seed)
				.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).iterations(1)
				.activation(Activation.RELU).weightInit(WeightInit.XAVIER).learningRate(rate)
				.updater(new Nesterovs(0.98)).regularization(true).l2(1e-4).list()
				.layer(0, new DenseLayer.Builder().nIn(width * height * channels).nOut(1000).build())
				.layer(1, new DenseLayer.Builder().nIn(1000).nOut(500).build())
				.layer(2, new DenseLayer.Builder().nIn(500).nOut(300).build())
				.layer(3, new DenseLayer.Builder().nIn(300).nOut(100).build())
				.layer(4,
						new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD).activation(Activation.SOFTMAX)
								.nIn(100).nOut(numClasses).build())
				.pretrain(false).backprop(true).setInputType(InputType.convolutional(height, width, channels)).build();

		MultiLayerNetwork network = new MultiLayerNetwork(conf);
		network.init();
		network.setListeners(new ScoreIterationListener(50));

		log.info("Network summary:");
		log.info(network.summary());
		StopWatch stopWatch = new StopWatch();
		stopWatch.start();
		log.info("Fitting....");
		for (int i = 0; i < numEpochs; i++) {
			log.info("Epoch {}", i);
			network.fit(trainDataIter);
		}
		stopWatch.stop();
		log.info("Fit complete. ELapsed time: [{}]", stopWatch.toString());
		evaluateNetwork(network);
		saveNetwork(network, new File(COMPUTATION_GRAPH_FILE_NAME));
		return network;

	}

	private void validateSingleImage(MultiLayerNetwork network) {
		File dir = new File(System.getProperty("user.home"), "/popsugar/shopstyle/core/train/sweaters");
		File file = new File(dir, "5015.jpg");
		log.info("Validating: {}", file.getAbsolutePath());
		NativeImageLoader loader = new NativeImageLoader(height, width, channels);
		try {
			INDArray image = loader.asMatrix(file);
			DataNormalization scaler = new VGG16ImagePreProcessor();
			scaler.transform(image);
			INDArray output = network.output(image);
			log.info("Predictions: {}", output);
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	private void loadData() {
		File trainDir = new File(System.getProperty("user.home"), "/data/dogscats/train");
		File testDir = new File(System.getProperty("user.home"), "/data/dogscats/valid");
		FileSplit trainFilesInDir = new FileSplit(trainDir, NativeImageLoader.ALLOWED_FORMATS, randNumGen);
		FileSplit testFilesInDir = new FileSplit(testDir, NativeImageLoader.ALLOWED_FORMATS, randNumGen);
		numClasses = trainDir.list().length;
		log.info("numClasses = {}", numClasses);
		ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();

		ImageRecordReader trainRecordReader = new ImageRecordReader(height, width, channels, labelMaker);
		ImageRecordReader testRecordReader = new ImageRecordReader(height, width, channels, labelMaker);

		try {
			trainRecordReader.initialize(trainFilesInDir);
			testRecordReader.initialize(testFilesInDir);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		RecordReaderDataSetIterator trainDataIter = new RecordReaderDataSetIterator(trainRecordReader, 64, 1,
				numClasses);
		DataNormalization trainScaler = new ImagePreProcessingScaler(0, 1);
		trainScaler.fit(trainDataIter);
		trainDataIter.setPreProcessor(trainScaler);
		this.trainDataIter = trainDataIter;

		RecordReaderDataSetIterator testDataIter = new RecordReaderDataSetIterator(testRecordReader, batchSize, 1,
				numClasses);
		DataNormalization testScaler = new ImagePreProcessingScaler(0, 1);
		testScaler.fit(testDataIter);
		testDataIter.setPreProcessor(testScaler);
		this.testDataIter = testDataIter;

	}

	private Optional<MultiLayerNetwork> loadComputationalGraph(File file) {
		MultiLayerNetwork pretrainedNet = null;
		if (file.exists()) {
			try {
				pretrainedNet = ModelSerializer.restoreMultiLayerNetwork(file);
			} catch (IOException e1) {
				e1.printStackTrace();
			}
		}
		return Optional.ofNullable(pretrainedNet);
	}

	private void saveNetwork(MultiLayerNetwork network, File file) {
		try {
			ModelSerializer.writeModel(network, file, true);
		} catch (IOException e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		}
	}

	private void evaluateNetwork(MultiLayerNetwork network) {
		log.info("Evaluate network....");
		Evaluation eval = new Evaluation(numClasses);
		while (testDataIter.hasNext()) {
			DataSet next = testDataIter.next();
			log.info("Labels: {}", next.getLabelNames());
			log.info("Labels: {}", next.getLabels());
			INDArray output = network.output(next.getFeatureMatrix()); // get the networks prediction
			log.info("Predictions: {}", next.getLabelNamesList());
			eval.eval(next.getLabels(), output); // check the prediction against the true class
		}

		log.info(eval.stats());
	}
}
