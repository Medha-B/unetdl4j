package org.sbml.spatial.segmentation;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;

import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.BaseImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.deeplearning4j.zoo.model.UNet;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.learning.config.Adam;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class crossVal {

	private static final Logger log = LoggerFactory.getLogger(crossVal.class);
	private static final int WIDTH = 128;
	private static final int HEIGHT = 128;
	private static final int CHANNELS = 3;

	public static void main(String[] args) {
		try {

			int batchSize = 1;
			String directory = System.getProperty("user.dir");
			String dataPath = directory + File.separator + "dataset";

			// String home = System.getProperty("user.dir");

			DataNormalization scaler = new ImagePreProcessingScaler(); // scale image between 0 and 1
			UnetPathLabelGenerator labeler = new UnetPathLabelGenerator();
			File rootDir = new File(dataPath + File.separator + "small_dataset");
			String[] allowedExtensions = BaseImageLoader.ALLOWED_FORMATS;
			Random rng = new Random();
			FileSplit fileSplit = new FileSplit(rootDir, allowedExtensions, rng);
			BalancedPathFilter pathFilter = new BalancedPathFilter(rng, allowedExtensions, labeler);

			// Since this data-set had 44 cell images and 44 corresponding ground truth
			// images, we can split for 4-fold cross validation (k=4 )
			int i, k = 4;
			double[] folds = new double[k];
			Arrays.fill(folds, 44 / k); // setting weights for splitting the input file
			InputSplit[] filesInDirSplit = fileSplit.sample(pathFilter, folds);

			/*
			 * Tried to obtained location of the split files ----> Did not know how to
			 * interpret the results for( i = 0; i < k; i++) { URI[] loc =
			 * filesInDirSplit[i].locations();
			 * System.out.println("Hopefully location for split " + i + " number");
			 * System.out.println(loc); long len = filesInDirSplit[i].length();
			 * System.out.println("Hopefully size for split" + i + "number");
			 * System.out.println(len); //This gave correct length File }
			 */

			log.info("*****SPLITTING DATASET******");
			// ImageRecordReader for each split input
			ImageRecordReader[] imageReader = new ImageRecordReader[k];
			for (i = 0; i < k; i++) {
				imageReader[i] = new ImageRecordReader(HEIGHT, WIDTH, CHANNELS, labeler);
				try {
					imageReader[i].initialize(filesInDirSplit[i]);
				} catch (IOException e) {
					e.printStackTrace();
				}
			}

			// RecordReaderDatasetIterator for each split input
			RecordReaderDataSetIterator[] set = new RecordReaderDataSetIterator[k];
			for (i = 0; i < k; i++) {
				set[i] = new RecordReaderDataSetIterator(imageReader[i], batchSize, 1, 1, true);
				scaler.fit(set[i]);
				set[i].setPreProcessor(scaler);

			}

			int numEpochs = 10;

			Map<Integer, Double> learningScheduleMap = new HashMap<>();
			learningScheduleMap.put(0, 0.00005);
			learningScheduleMap.put(200, 0.00001);
			learningScheduleMap.put(600, 0.000005);
			learningScheduleMap.put(800, 0.0000001);
//            learningScheduleMap.put(1000, 0.00001);

			// Training the model for cross validation
			int testFold = 0;
			while (testFold < k) {
				ComputationGraph model = UNet.builder().updater(new Adam(1e-4)).build().init(); // Initializing a new
																								// model
				model.addListeners(new ScoreIterationListener());
				// System.out.println("Initializing new model");
				for (i = 0; i < k; i++) {

					if (i == testFold) {
						continue; // Model is not trained on the testfold
					} else {
						System.out.println("fitting model"); // Model gets trained on all folds except the testFold
						model.fit(set[i], numEpochs);
					}

				}

				// Now saving the model weights so that it can be tested later with the
				// corresponding testfold. Need a method to save the testfold so that the images
				// and labels can be separated.
				File locationTosave = new File(directory + File.separator + "saveModel" + File.separator + "unetSave["
						+ testFold + "]" + ".zip"); // So that I know which testFold to test this model against
				// File locationTosave = new File(home + File.separator + "unetSave[" + testFold
				// + "]" + ".zip");
				boolean saveUpdater = false;
				ModelSerializer.writeModel(model, locationTosave, saveUpdater);
				testFold++;
			}

		} catch (Exception e) {
			System.err.println("Oooooops");
			e.printStackTrace();
		}

	}

}
