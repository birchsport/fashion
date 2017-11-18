package com.shopstyle.dl.fashion.configuration;

import java.io.File;
import java.io.IOException;
import java.util.Random;

import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.BaseImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.zoo.PretrainedType;
import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.zoo.model.VGG16;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

@Configuration
public class DL4JConfiguration {

	private static final String[] allowedExtensions = BaseImageLoader.ALLOWED_FORMATS;

	private static final long seed = 12345;

	private static final Random randNumGen = new Random(seed);

	private static final int height = 224;
	private static final int width = 224;
	private static final int channels = 3;

	@Bean
	public Model vgg16() {
		final int numRows = 28;
		final int numColumns = 28;
		int outputNum = 10; // number of output classes
		int batchSize = 64; // batch size for each epoch
		int rngSeed = 123; // random number seed for reproducibility
		int numEpochs = 15; // number of epochs to perform
		double rate = 0.0015; // learning rate
		int numClasses = 5;

		FineTuneConfiguration fineTuneConf = new FineTuneConfiguration.Builder().learningRate(5e-5)
				.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).updater(Updater.NESTEROVS)
				.seed(rngSeed).build();

		ZooModel zooModel = new VGG16();
		ComputationGraph pretrainedNet = null;
		try {
			pretrainedNet = (ComputationGraph) zooModel.initPretrained(PretrainedType.IMAGENET);
		} catch (IOException e) {
			throw new IllegalStateException("Unable to load pretrained model");
		}
		pretrainedNet.init();

		ComputationGraph vgg16Transfer = new TransferLearning.GraphBuilder(pretrainedNet)
				.fineTuneConfiguration(fineTuneConf).setFeatureExtractor("fc2")
				.removeVertexKeepConnections("predictions")
				.addLayer("predictions",
						new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD).nIn(4096)
								.nOut(numClasses).weightInit(WeightInit.XAVIER).activation(Activation.SOFTMAX).build(),
						"fc2")
				.build();

		RecordReaderDataSetIterator rrdi = new RecordReaderDataSetIterator(loadData(), 100, 1, 2);

		System.out.println("Fitting....");
		pretrainedNet.fit(rrdi);
		System.out.println("Fit.");

		return pretrainedNet;

	}

	private static RecordReader loadData() {
		FileSplit split = new FileSplit(new File(System.getProperty("user.home"), "/data/dogscats/train"));
        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
        ImageRecordReader recordReader = new ImageRecordReader(height,width,channels,labelMaker);
        try {
			recordReader.initialize(split);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
        return recordReader;


	}
}
