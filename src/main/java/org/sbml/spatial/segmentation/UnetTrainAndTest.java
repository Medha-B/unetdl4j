package org.sbml.spatial.segmentation;

import java.io.File;
import java.util.HashMap;
import java.util.Map;

import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.zoo.model.UNet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Adam;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

//This class is for training the UNet model on any number of images and then testing it.
public class UnetTrainAndTest {

	private static final Logger log = LoggerFactory.getLogger(UnetTrainAndTest.class);

	public static void main(String[] args) {
		try {
			int batchSize = 10;
			// String home = System.getProperty("user.home");

			String directory = System.getProperty("user.dir");
			String dataPath = directory + File.separator + "dataset";

			// This is for setting the path to test file
			String pathToImage;
			if (args.length > 0) {
				pathToImage = args[0];
			} else {
				pathToImage = dataPath + File.separator + "raw_images" + File.separator + "F01_621w1_crop13.tif";
			}

			File rootDir = new File(dataPath + File.separator + "small_dataset");
			PreProcess prep = new PreProcess(rootDir, batchSize);
			DataSetIterator imageDataSetIterator = prep.dataProcessed();

			Map<Integer, Double> learningScheduleMap = new HashMap<>();
			learningScheduleMap.put(0, 0.00005);
			learningScheduleMap.put(200, 0.00001);
			learningScheduleMap.put(600, 0.000005);
			learningScheduleMap.put(800, 0.0000001);
//	            learningScheduleMap.put(1000, 0.00001);

			int numEpochs = 20;

			log.info("*****TRAIN MODEL******");
			ComputationGraph model = UNet.builder().updater(new Adam(1e-4)).build().init();
//	            ComputationGraph model  = UNet.builder().updater(new Adam(new MapSchedule(ScheduleType.ITERATION, learningScheduleMap))).build().init();
			model.addListeners(new ScoreIterationListener());
			model.fit(imageDataSetIterator, numEpochs);

			log.info("*****EVALUATE MODEL******");
			Inference infer = new Inference(model, pathToImage, directory);
			infer.imgOut();

		} catch (Exception e) {
			System.err.println("Oooooops");
			e.printStackTrace();
		}
	}
}
