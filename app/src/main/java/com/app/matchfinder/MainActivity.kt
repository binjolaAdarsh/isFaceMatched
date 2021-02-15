package com.app.matchfinder

import android.app.Activity
import android.content.Intent
import android.graphics.Bitmap
import android.os.Bundle
import android.provider.MediaStore
import android.util.Log
import android.view.View
import android.view.WindowManager
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import com.google.firebase.ml.vision.common.FirebaseVisionImage
import com.google.firebase.ml.vision.common.FirebaseVisionImageMetadata
import com.google.firebase.ml.vision.face.FirebaseVisionFace
import com.theartofdev.edmodo.cropper.CropImage
import com.theartofdev.edmodo.cropper.CropImageView
import kotlinx.android.synthetic.main.activity_main.*
import java.io.IOException



class MainActivity : AppCompatActivity() {
    var firstImageAdded = false
    var secondImageAdded = false
    var firstImage = false
    var firstFaceFound=false
    var secondFaceFound=false
    var firstImageBitmap: Bitmap? = null
    var secondImageBitmap: Bitmap? = null
    private val helper: DetectHelper by lazy {
        DetectHelper(this@MainActivity)
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        fab1.setOnClickListener {
            firstImage = true
            startCropScreen()   // open camera
        }
        fab2.setOnClickListener {
            firstImage = false
          startCropScreen() // open camera
        }

        btFindMatch.setOnClickListener {

                if(!firstImageAdded){
                    "Please add first image.".toast(this@MainActivity)
                    return@setOnClickListener
                }
                if(!secondImageAdded){
                    "Please add second image.".toast(this@MainActivity)
                    return@setOnClickListener
                }
                if(!firstFaceFound){
                    "no face found on image 1".toast(this@MainActivity)
                    return@setOnClickListener
                }
                if(!secondFaceFound){
                    "no face found on image 2".toast(this@MainActivity)
                    return@setOnClickListener
                }

                var highestSimilarityScore = -1f
                var result: String
                val p = helper.findNearest(helper.firstEmbedding,helper.secondEmbedding)


            // if the p value is greater then -1 and less than 1 then the face in both images is similar.
            // if its greater then 1 then its considered as not matched.
              result =  if ( p > highestSimilarityScore && p < 1f) {
                    highestSimilarityScore = p
                 "Both person MATCHED"
                }else{
                   "Person didn't MATCHED"
                }

                val format = String.format("%.2f", highestSimilarityScore)
                var percentage=(100 - (format.toFloat() * 100f))
                if(percentage>100)
                    percentage=0.0f

                AlertDialog.Builder(this).setTitle(result).setMessage("Percentage after matched : $percentage")
                    .setPositiveButton("ok") { _, _-> }.show()
     }



        helper.setupDetectorOption()
    }


    private fun startCropScreen() {
        CropImage.activity()
            .setGuidelines(CropImageView.Guidelines.ON)
            .start(this)
    }


    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode == CropImage.CROP_IMAGE_ACTIVITY_REQUEST_CODE) {
            val result = CropImage.getActivityResult(data)
            if (resultCode == Activity.RESULT_OK) {
                val resultUri = result.uri
                val selectedBitmap= MediaStore.Images.Media.getBitmap(this.contentResolver, resultUri)
                if (firstImage) { // result for the first image
                    // show image to ui
                    BaseApplication.getGlide().load(resultUri).into(ivFirst)
                    firstImageAdded = true
                    firstImageBitmap =selectedBitmap

                } else {
                    BaseApplication.getGlide().load(resultUri).into(ivSecond)
                    secondImageAdded = true
                    secondImageBitmap = selectedBitmap
                }

                showProgress(true)
                val image: FirebaseVisionImage
                try {
                    image = FirebaseVisionImage.fromByteArray(bitmapToNV21(selectedBitmap),
                        getVisionMetaData(selectedBitmap))
                    helper.detectFace(image,onfaceDetected,errorWhileDetectingFace)
                } catch (e: IOException) {
                    showProgress(false)
                   Log.e(TAG,"exception while getting the vision image from file path")
                }
            } else if (resultCode == CropImage.CROP_IMAGE_ACTIVITY_RESULT_ERROR_CODE) {
                val error = result.error
            }
        }
    }

    private fun getVisionMetaData(selectedBitmap: Bitmap)=
         FirebaseVisionImageMetadata.Builder()
            .setWidth(selectedBitmap.width)
            .setHeight(selectedBitmap.height)
            .setFormat(FirebaseVisionImageMetadata.IMAGE_FORMAT_NV21)
            .setRotation(FirebaseVisionImageMetadata.ROTATION_0)
            .build()



    var onfaceDetected :(MutableList<FirebaseVisionFace>?)->Unit={
        showProgress(false)
        it?.let { faces ->
            if (faces.isEmpty()) {
                "No face Found.Please try again".toast(this@MainActivity)
                if (firstImage) firstFaceFound=false else secondFaceFound=false
            }else{
                "Found ${faces.size} faces ".toast(this@MainActivity)


                if (firstImage) {
                    helper.createFirstEmbedding() // resetting the embedding

                    // getting first embedding
                    helper.firstEmbedding =
                        helper.getFaceEmbedding(firstImageBitmap!!,faces[0].boundingBox)
                    firstFaceFound=true
                }else{
                    helper.createSecondEmbedding() // resetting the embedding

                    // getting the second embedding
                    helper.secondEmbedding =
                        helper.getFaceEmbedding(secondImageBitmap!!,faces[0].boundingBox)
                    secondFaceFound=true

                }
            }
        }
    }

    var errorWhileDetectingFace :(Exception)->Unit={
        showProgress(false)
        Log.e(TAG,"Error while detecting the face")
    }

  companion object {
        private const val TAG = "MainActivity"
  }




    private fun showProgress(flag:Boolean){
        if(flag){ // show progress make ui not touchable
            window.setFlags(
                WindowManager.LayoutParams.FLAG_NOT_TOUCHABLE,
                WindowManager.LayoutParams.FLAG_NOT_TOUCHABLE)
            progressBar.visibility= View.VISIBLE

        }else{ // hide progress make ui touchable
            window.clearFlags(WindowManager.LayoutParams.FLAG_NOT_TOUCHABLE)

            progressBar.visibility= View.GONE
        }

    }



}
