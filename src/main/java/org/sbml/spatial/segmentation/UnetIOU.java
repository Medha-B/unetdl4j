package org.sbml.spatial.segmentation;

import java.awt.image.BufferedImage;
import java.io.File;

import javax.imageio.ImageIO;

import org.datavec.image.loader.NativeImageLoader;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.indexing.conditions.Conditions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * The class UnetIOU.
 * 
 * This class is used to obtain IOU values and Dice Coefficient for images
 * inferred from the U-Net model . Date Created: July 9, 2020
 * 
 * @author Medha Bhattacharya
 * @author Akira Funahashi
 * @author Yuta Tokuoka
 * @author Kaito Ii
 *
 */

public class UnetIOU {
	private static final Logger log = LoggerFactory.getLogger(UnetIOU.class);
	/**
	 * int Width the width of input image set to 128
	 */
	private static final int WIDTH = 128;
	/**
	 * int HEIGHT the height of input image set to 128
	 */
	private static final int HEIGHT = 128;
	/**
	 * int Channels the number of input channels in image set to 3
	 */
	private static final int CHANNELS = 3;

	public static void main(String[] args) {
		try {

			// String home = System.getProperty("user.home");
			String directory = System.getProperty("user.dir");
			String dataPath = directory + File.separator + "dataset";

			// This is for setting the path to ground truth image
			String pathToGroundTruth;
			if (args.length > 0) {
				pathToGroundTruth = args[0];
			} else {
				pathToGroundTruth = dataPath + File.separator + "Inference(300x100)" + File.separator + "gt_20.tif";
			}

			// This is for setting the path to inferred image
			String pathToInferredImage;
			if (args.length > 0) {
				pathToInferredImage = args[0];
			} else {
				pathToInferredImage = dataPath + File.separator + "Inference(300x100)" + File.separator + "inf_20.tif";
			}

			log.info("*****QUANTITATIVE EVALUATION******");
			NativeImageLoader loader = new NativeImageLoader(HEIGHT, WIDTH, CHANNELS);
			BufferedImage groundTruth = ImageIO.read(new File(pathToGroundTruth));
			INDArray gTruth = loader.asMatrix(groundTruth);

			/*
			 * System.out.println("Ground Truth"); System.out.println(gTruth);
			 * 
			 * //For finding the total number of elements in the Ground Truth Image long
			 * elementsGT = gTruth.length();
			 * 
			 * System.out.println("Total elements in ground truth");
			 * System.out.println(elementsGT);
			 * 
			 * //To verify whether the scan function works correctly or not int verifyGT =
			 * gTruth.scan(Conditions.greaterThanOrEqual(0.0)).intValue();
			 * 
			 * System.out.
			 * println("Number of elements in ground truth greater than or equal to 0");
			 * System.out.println(verifyGT);
			 * 
			 * int countGT = gTruth.scan(Conditions.greaterThan(0.0)).intValue();
			 * 
			 * System.out.println("Number of elements in ground truth greater than 0");
			 * System.out.println(countGT);
			 */

			BufferedImage outImage = ImageIO.read(new File(pathToInferredImage));
			INDArray inferred = loader.asMatrix(outImage);

			/*
			 * System.out.println("Inferred Image"); System.out.println(inferred);
			 * 
			 * long elementsInf = inferred.length();
			 * 
			 * System.out.println("Total elements in inferred image");
			 * System.out.println(elementsInf);
			 * 
			 * int verifyInf = inferred.scan(Conditions.greaterThanOrEqual(0.0)).intValue();
			 * 
			 * System.out.
			 * println("Number of elements in inferred image greater than or equal to 0");
			 * System.out.println(verifyInf);
			 * 
			 * int countInf = inferred.scan(Conditions.greaterThan(0.0)).intValue();
			 * 
			 * System.out.println("Number of elements in inferred image greater than 0");
			 * System.out.println(countInf);
			 */
			INDArray resultAdd = gTruth.add(inferred);
			/*
			 * System.out.println("Addition of Ground Truth and Inferred Image");
			 * System.out.println(resultAdd);
			 */

			int union = resultAdd.scan(Conditions.greaterThan(0.0)).intValue();

			/*
			 * System.out.println("Number of elements in Union greater than 0");
			 * System.out.println(union);
			 */

			INDArray resultMul = gTruth.mul(inferred);
			/*
			 * System.out.println("Multiplication of Ground Truth and Inferred Image");
			 * System.out.println(resultMul);
			 */

			int intersection = resultMul.scan(Conditions.greaterThan(0.0)).intValue();

			/*
			 * System.out.println("Number of elements in Intersection greater than 0");
			 * System.out.println(intersection);
			 */

			float iou = (float) intersection / union;

			System.out.println("Intersection over Union");
			System.out.println(iou);

			// Calculating the Dice coefficient
			int countGT = gTruth.scan(Conditions.greaterThan(0.0)).intValue();
			int countInf = inferred.scan(Conditions.greaterThan(0.0)).intValue();
			// long elementsGT = gTruth.length();
			// long elementsInf = inferred.length();
			// long total = elementsGT + elementsInf;
			int total = countGT + countInf;
			float Dice = (float) 2 * intersection / total;
			// float Dice = (2*iou)/(iou+1);

			System.out.println("Dice Coefficient");
			System.out.println(Dice);

		} catch (Exception e) {
			System.err.println("Oooooops");
			e.printStackTrace();
		}
	}
}
