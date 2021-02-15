package com.app.matchfinder

import android.content.Context
import android.content.res.AssetManager
import android.graphics.*
import android.util.Log
import com.google.firebase.ml.vision.FirebaseVision
import com.google.firebase.ml.vision.common.FirebaseVisionImage
import com.google.firebase.ml.vision.face.FirebaseVisionFace
import com.google.firebase.ml.vision.face.FirebaseVisionFaceDetectorOptions
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.sqrt

class DetectHelper(context: Context) {


    private val intValues: IntArray = IntArray(TF_OD_API_INPUT_SIZE * TF_OD_API_INPUT_SIZE)

    private lateinit var options: FirebaseVisionFaceDetectorOptions
    private val interpreter: Interpreter

    init {
        val interpreterOptions = Interpreter.Options().apply {
            setNumThreads(4)
        }
        interpreter = Interpreter(
            FileUtil.loadMappedFile(context, "mobile_face_net.tflite"),
            interpreterOptions
        )

    }



    fun setupDetectorOption() {
        options =
            FirebaseVisionFaceDetectorOptions.Builder()
                .setPerformanceMode(FirebaseVisionFaceDetectorOptions.ACCURATE)
                .setLandmarkMode(FirebaseVisionFaceDetectorOptions.NO_LANDMARKS)
                .setClassificationMode(FirebaseVisionFaceDetectorOptions.NO_CLASSIFICATIONS)
                .build()
    }

    fun detectFace(
        image: FirebaseVisionImage, success: (MutableList<FirebaseVisionFace>?) -> Unit,
        failure: (Exception) -> Unit
    ) {
        FirebaseVision.getInstance()
            .getVisionFaceDetector(options).detectInImage(image)
            .addOnSuccessListener {
                success(it)
            }
            .addOnFailureListener {
                // Task failed with an exception
                // ...
                failure(it)
            }


    }



    private fun createAndDrawCanvas(
        imageBitmap: Bitmap,
        tfOdApiInputSize: Int,
        boundingBox: Rect
    ): Bitmap {
        var faceBmp = Bitmap.createBitmap(
            tfOdApiInputSize, tfOdApiInputSize,
            Bitmap.Config.ARGB_8888
        )
        val cvFace = Canvas(faceBmp)
        // maps original coordinates to portrait coordinates
        val faceBB = RectF(boundingBox)
        val sx: Float =
            (tfOdApiInputSize.toFloat()) / faceBB.width()
        val sy: Float =
            (tfOdApiInputSize.toFloat()) / faceBB.height()
        val matrix = Matrix()
        matrix.postTranslate(-faceBB.left, -faceBB.top)
        matrix.postScale(sx, sy)
        cvFace.drawBitmap(imageBitmap, matrix, null)
        return faceBmp
    }




    lateinit var firstEmbedding: FloatArray
    lateinit var secondEmbedding: FloatArray
    fun createFirstEmbedding() {
        firstEmbedding = FloatArray(128)
    }

    fun createSecondEmbedding() {
        secondEmbedding = FloatArray(128)
    }


    fun getFaceEmbedding(firstImageBitmap: Bitmap, boundingBox: Rect): FloatArray {
        return runFaceNet(
            convertBitmapToBuffer(
                createAndDrawCanvas(firstImageBitmap, TF_OD_API_INPUT_SIZE, boundingBox)
            )
        )[0]
    }

    private fun convertBitmapToBuffer(bitmap: Bitmap): ByteBuffer {
        // Preprocess the image data from 0-255 int to normalized float based
// on the provided parameters.
        bitmap.getPixels(
            intValues,
            0,
            bitmap.width,
            0,
            0,
            bitmap.width,
            bitmap.height
        )
        var byteArray =
            ByteBuffer.allocateDirect(1 * TF_OD_API_INPUT_SIZE * TF_OD_API_INPUT_SIZE * 3 * 4)
                .apply {
                    order(ByteOrder.nativeOrder())
                }
        byteArray.rewind()
        for (i in 0 until TF_OD_API_INPUT_SIZE) {
            for (j in 0 until TF_OD_API_INPUT_SIZE) {
                val pixelValue = intValues[i * TF_OD_API_INPUT_SIZE + j]

                byteArray.putFloat(((pixelValue shr 16 and 0xFF) - IMAGE_MEAN) / IMAGE_STD)
                byteArray.putFloat(((pixelValue shr 8 and 0xFF) - IMAGE_MEAN) / IMAGE_STD)
                byteArray.putFloat(((pixelValue and 0xFF) - IMAGE_MEAN) / IMAGE_STD)
            }
        }
        return byteArray
    }

    private fun runFaceNet(inputs: ByteBuffer): Array<FloatArray> {

        val t1 = System.currentTimeMillis()
        val outputs = Array(1) { FloatArray(192) }
        interpreter.run(inputs, outputs)
        Log.i("Performance", "FaceNet Inference Speed in ms : ${System.currentTimeMillis() - t1}")
        return outputs
    }

    fun findNearest(emb1: FloatArray, emb2: FloatArray): Float {

        var distance = 0f
        for (i in emb1.indices) {
            val diff = emb1[i] - emb2[i]
            distance += diff * diff
        }
        distance = sqrt(distance.toDouble()).toFloat()
        return distance

    }
}