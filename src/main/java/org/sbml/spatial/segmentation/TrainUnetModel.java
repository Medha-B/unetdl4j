package org.sbml.spatial.segmentation;

import java.awt.image.BufferedImage;
import java.io.File;

import javax.imageio.ImageIO;

import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
//import org.deeplearning4j.zoo.ZooModel;

/**
 * The class TrainUnetModel.
 * 
 * This class is used for training the U-Net model for single input channel.
 * Date Created: July 5, 2020
 * 
 * @author Medha Bhattacharya
 * @author Akira Funahashi
 * @author Yuta Tokuoka
 * @author Kaito Ii
 *
 */

public class TrainUnetModel {

	private static final Logger log = LoggerFactory.getLogger(TrainUnetModel.class);

	public static void main(String[] args) {
		try {
			int batchSize = 10;
			// String home = System.getProperty("user.home");

			String directory = System.getProperty("user.dir");
			String dataPath = directory + File.separator + "dataset";

			String pathToImage;
			if (args.length > 0) {
				pathToImage = args[0];
			} else {
				pathToImage = dataPath + File.separator + "raw_images" + File.separator + "F01_621w1_crop13.tif";
			}

			File rootDir = new File(dataPath + File.separator + "small_dataset");
			PreProcess prep = new PreProcess(rootDir, batchSize);
			DataSetIterator imageDataSetIterator = prep.dataProcessed();

			int numEpochs = 1;

			log.info("*****TRAIN MODEL******");
			Fitter fit = new Fitter(imageDataSetIterator, numEpochs);
			ComputationGraph model = fit.FitModel();

			log.warn(model.summary());

			log.info("*****EVALUATE MODEL******");
			Inference infer = new Inference(model, pathToImage, directory);
			BufferedImage bufimg = infer.imgOut();
			ImageIO.write(bufimg, "tif", new File(directory + File.separator + "outputUnet.tif"));

		} catch (Exception e) {
			System.err.println("Oooooops");
			e.printStackTrace();
		}
	}
}
