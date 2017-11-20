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
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.deeplearning4j.zoo.PretrainedType;
import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.zoo.model.VGG16;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.util.StopWatch;

@Configuration
public class PretrainedConfiguration {

	private static final Logger log = org.slf4j.LoggerFactory.getLogger(PretrainedConfiguration.class);
	private static final String COMPUTATION_GRAPH_FILE_NAME = "fashion_comp_graph.zip";

	private static final long seed = 12345;

	private static final Random randNumGen = new Random(seed);

	// private static final int height = 164;
	// private static final int width = 205;
	private static final int height = 224;
	private static final int width = 224;
	private static final int channels = 3;
	private static final int batchSize = 8;
	private static final int numEpochs = 1;
	private int numClasses = 2;
	private RecordReaderDataSetIterator trainDataIter;
	private RecordReaderDataSetIterator testDataIter;
	private static final String featureExtractionLayer = "block5_pool";

	@Bean
	public ComputationGraph createComputationGraph() {
		double rate = 0.001; // learning rate
		loadData();
		Optional<ComputationGraph> optional = loadGraph(new File(COMPUTATION_GRAPH_FILE_NAME));
		if (optional.isPresent()) {
			log.info("Loaded pretrained model:");
			ComputationGraph network = optional.get();
			log.info(network.summary());
			// validateSingleImage(network);
			evaluateNetwork(network);
			return network;
		}

		log.info("Building model....");
		ComputationGraph graph = buildPretrainedGraph(rate);
		// graph.setListeners(new ScoreIterationListener(5));
		UIServer uiServer = UIServer.getInstance();

		// Configure where the network information (gradients, activations, score vs.
		// time etc) is to be stored
		// Then add the StatsListener to collect this information from the network, as
		// it trains
		StatsStorage statsStorage = new InMemoryStatsStorage(); // Alternative: new FileStatsStorage(File) - see
																// UIStorageExample
		int listenerFrequency = 1;
		graph.setListeners(new StatsListener(statsStorage, listenerFrequency), new ScoreIterationListener(5));

		// Attach the StatsStorage instance to the UI: this allows the contents of the
		// StatsStorage to be visualized
		uiServer.attach(statsStorage);

		log.info("Graph summary:");
		log.info(graph.summary());
		StopWatch stopWatch = new StopWatch();
		stopWatch.start();
		log.info("Fitting....");
		for (int i = 0; i < numEpochs; i++) {
			log.info("Epoch {}", i);
			graph.fit(trainDataIter);
		}
		stopWatch.stop();
		log.info("Fit complete. ELapsed time: [{}]", stopWatch.toString());
		evaluateNetwork(graph);
		saveNetwork(graph, new File(COMPUTATION_GRAPH_FILE_NAME));
		return graph;

	}

	private ComputationGraph buildPretrainedGraph(double rate) {
		FineTuneConfiguration fineTuneConf = new FineTuneConfiguration.Builder().activation(Activation.LEAKYRELU)
				.weightInit(WeightInit.RELU).learningRate(5e-5)
				.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).updater(Updater.NESTEROVS)
				.dropOut(0.5).seed(seed).build();

		ZooModel zooModel = new VGG16();
		ComputationGraph pretrainedNet;
		try {
			pretrainedNet = (ComputationGraph) zooModel.initPretrained(PretrainedType.IMAGENET);
		} catch (IOException e) {
			throw new IllegalStateException("Unable to load pretrained model");
		}
		pretrainedNet.init();

		ComputationGraph vgg16Transfer = new TransferLearning.GraphBuilder(pretrainedNet)
				.fineTuneConfiguration(fineTuneConf).setFeatureExtractor(featureExtractionLayer) // "block5_pool"
																									// and
																									// below are
																									// frozen
				.nOutReplace("fc2", 1024, WeightInit.XAVIER) // modify nOut of the "fc2" vertex
				.removeVertexAndConnections("predictions") // remove the final vertex and it's connections
				.addLayer("fc3", new DenseLayer.Builder().activation(Activation.TANH).nIn(1024).nOut(256).build(),
						"fc2") // add in a new dense layer
				.addLayer("newpredictions",
						new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
								.activation(Activation.SOFTMAX).nIn(256).nOut(numClasses).build(),
						"fc3") // add in a final output dense layer,
								// note that learning related configurations applied on a new layer here will be
								// honored
								// In other words - these will override the finetune confs.
								// For eg. activation function will be softmax not RELU
				.setOutputs("newpredictions") // since we removed the output vertex and it's connections we need to
												// specify outputs for the graph
				.build();
		return vgg16Transfer;
	}

	private void loadData() {
		File trainDir = new File(System.getProperty("user.home"), "/data/dogscats/train");
		File testDir = new File(System.getProperty("user.home"), "/data/dogscats/valid");
		FileSplit trainFilesInDir = new FileSplit(trainDir, NativeImageLoader.ALLOWED_FORMATS, randNumGen);
		FileSplit testFilesInDir = new FileSplit(testDir, NativeImageLoader.ALLOWED_FORMATS, randNumGen);
		numClasses = trainDir.list().length;
		log.info("numClasses = {}", numClasses);

		ImageRecordReader trainRecordReader = new ImageRecordReader(height, width, channels,
				new ParentPathLabelGenerator());
		ImageRecordReader testRecordReader = new ImageRecordReader(height, width, channels,
				new ParentPathLabelGenerator());

		try {
			trainRecordReader.initialize(trainFilesInDir);
			testRecordReader.initialize(testFilesInDir);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		RecordReaderDataSetIterator trainDataIter = new RecordReaderDataSetIterator(trainRecordReader, batchSize, 1,
				numClasses);
		DataNormalization trainScaler = new ImagePreProcessingScaler(0, 1);
		// trainScaler.fit(trainDataIter);
		// trainDataIter.setPreProcessor(trainScaler);
		this.trainDataIter = trainDataIter;

		RecordReaderDataSetIterator testDataIter = new RecordReaderDataSetIterator(testRecordReader, 4, 1, numClasses);
		DataNormalization testScaler = new ImagePreProcessingScaler(0, 1);
		// testScaler.fit(testDataIter);
		// testDataIter.setPreProcessor(testScaler);
		this.testDataIter = testDataIter;

	}

	private Optional<ComputationGraph> loadGraph(File file) {
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

	private void saveNetwork(ComputationGraph network, File file) {
		try {
			ModelSerializer.writeModel(network, file, true);
		} catch (IOException e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		}
	}

	private void evaluateNetwork(ComputationGraph network) {
		log.info("Evaluate network....");
		Evaluation eval = new Evaluation(numClasses);
		while (testDataIter.hasNext()) {
			DataSet next = testDataIter.next();
			// log.info("Labels: {}", next.getLabels());
			INDArray[] output = network.output(next.getFeatureMatrix()); // get the networks prediction
			// log.info("Predictions: {}", output);
			eval.eval(next.getLabels(), output[0]); // check the prediction against the true class
		}

		log.info(eval.stats());
	}
}
