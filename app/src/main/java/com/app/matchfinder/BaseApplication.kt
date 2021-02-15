package com.app.matchfinder

import android.app.Application
import com.bumptech.glide.Glide
import com.bumptech.glide.RequestManager
import com.bumptech.glide.request.RequestOptions

class BaseApplication : Application() {
    override fun onCreate() {
        super.onCreate()
        val provideRequestOptions = provideRequestOptions()

        glideInstance = provideGlideInstance(this, provideRequestOptions)
    }

    private fun provideRequestOptions(): RequestOptions {
        return RequestOptions.placeholderOf(R.drawable.ic_image_black_24dp)
            .error(R.drawable.ic_image_black_24dp)
    }

    private fun provideGlideInstance(application: Application, requestOptions: RequestOptions) =
        Glide.with(application).setDefaultRequestOptions(requestOptions)

    companion object {
        private lateinit var glideInstance: RequestManager

        fun getGlide() = glideInstance
    }
}