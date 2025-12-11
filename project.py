

<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="description"
        content="Pansharpening Diffusion model.">
  <meta name="keywords" content="Super-Resolution, Pansharpening, Diffusion, Knowledge distillation">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>U-Know-DiffPAN: An Uncertainty-aware Knowledge Distillation Diffusion Framework with Details Enhancement for PAN-Sharpening</title>
  <style>
    @keyframes shake {
      0% {transform: translateX(0);}
      10%, 90% {transform: translateX(-5px);}
      20%, 80% {transform: translateX(5px);}
      30%, 50%, 70% {transform: translateX(-5px);}
      40%, 60% {transform: translateX(5px);}
      100% {transform: translateX(0);}
    }

    #blur {
      /* font-size: 60px; */
      color: rgb(52,144,197);
      filter: blur(3x);
      -webkit-filter: blur(2.5px);
      animation: shake 1s infinite;
      display: inline-block;;
      clear: none;
    }
    #shake {
      /* font-size: 60px; */
      animation: shake 1s infinite;
      display: inline-block;;
      clear: none;
    }
  </style>

  <!-- Google tag (gtag.js) -->
  <script async src="https://www.googletagmanager.com/gtag/js?id=G-HR3PWVQLH2"></script>
  <script>
    window.dataLayer = window.dataLayer || [];
    function gtag(){dataLayer.push(arguments);}
    gtag('js', new Date());
    gtag('config', 'G-HR3PWVQLH2');
  </script>

  <!-- <link href="https://fonts.googleapis.com/css?family=Google+Sans|Noto+Sans|Castoro"
        rel="stylesheet">

  <link rel="stylesheet" href="./static/css/bulma.min.css">
  <link rel="stylesheet" href="./static/css/bulma-carousel.min.css">
  <link rel="stylesheet" href="./static/css/bulma-slider.min.css">
  <link rel="stylesheet" href="./static/css/fontawesome.all.min.css">
  <link rel="stylesheet"
        href="https://cdn.jsdelivr.net/gh/jpswalsh/academicons@1/css/academicons.min.css">
  <link rel="stylesheet" href="./static/css/index.css">
    
  <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.5.1/jquery.min.js"></script>
  <script defer src="./static/js/fontawesome.all.min.js"></script>
  <script src="./static/js/bulma-carousel.min.js"></script>
  <script src="./static/js/bulma-slider.min.js"></script>
  <script src="./static/js/index.js"></script>
  <script src="./static/js/video_comparison.js"></script> -->
    
  <link href="https://fonts.googleapis.com/css?family=Google+Sans|Noto+Sans|Castoro"
  rel="stylesheet">
  
  <link rel="stylesheet" href="./static/css/bulma.min.css">
  <link rel="stylesheet" href="./static/css/bulma-carousel.min.css">
  <link rel="stylesheet" href="./static/css/bulma-slider.min.css">
  <link rel="stylesheet" href="./static/css/fontawesome.all.min.css">
  <link rel="stylesheet"
  href="https://cdn.jsdelivr.net/gh/jpswalsh/academicons@1/css/academicons.min.css">
  <link rel="stylesheet" href="./static/css/index.css">
  <!-- <link rel="icon" href="./images/logo.svg"> -->
  <link rel="icon" sizes="256x256" href="./static/image/globe.png" type="image/png"/>

  <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.5.1/jquery.min.js"></script>
  <script defer src="./static/js/fontawesome.all.min.js"></script>
  <script src="./static/js/bulma-carousel.min.js"></script>
  <script src="./static/js/bulma-slider.min.js"></script>
  <script src="./static/js/index.js"></script>

  <!-- Vendor Stylesheets -->
  <!--=================js==========================-->
  <link rel="stylesheet" href="./static/css/tab_gallery.css">
  <link rel="stylesheet" href="./static/css/juxtapose.css">
  <!-- <script src="./static/js/index.js"></script> -->
  <!-- <link rel="stylesheet" href="./menlo"> -->
  <link rel="stylesheet" href="./static/css/image_card_fader.css">
  <link rel="stylesheet" href="./static/css/image_card_slider.css">  
</head>
<body>

<nav class="navbar" role="navigation" aria-label="main navigation">
  <div class="navbar-brand">
    <a role="button" class="navbar-burger" aria-label="menu" aria-expanded="false">
      <span aria-hidden="true"></span>
      <span aria-hidden="true"></span>
      <span aria-hidden="true"></span>
    </a>
  </div>
  <!--<div class="navbar-menu">
    <div class="navbar-start" style="flex-grow: 1; justify-content: center;">
      <a target="_blank" rel="noopener noreferrer" class="navbar-item" href="https://www.viclab.kaist.ac.kr">
      <span class="icon">
          <i class="fas fa-home"></i>
      </span>
      </a>

      <div class="navbar-item has-dropdown is-hoverable">
        <a class="navbar-link">
          Related Works
        </a>
        <div class="navbar-dropdown">
          <a target="_blank" rel="noopener noreferrer" class="navbar-item" href="https://hangg7.com/dycheck/">
            Dycheck (Neurips 2022)
          </a>
          <a target="_blank" rel="noopener noreferrer" class="navbar-item" href="https://robust-dynrf.github.io/">
            RoDynRF (CVPR 2023)
          </a>
          <a target="_blank" rel="noopener noreferrer" class="navbar-item" href="https://caoang327.github.io/HexPlane/">
            HexPlane (CVPR 2023)
          </a>
          <a target="_blank" rel="noopener noreferrer" class="navbar-item" href="https://wangpeng000.github.io/BAD-NeRF/">
            BAD-NeRF (CVPR 2023)
          </a>
          <a target="_blank" rel="noopener noreferrer" class="navbar-item" href="https://dogyoonlee.github.io/dpnerf/">
            DP-NeRF (CVPR 2023)
          </a>
          <a target="_blank" rel="noopener noreferrer" class="navbar-item" href="https://jaminfong.cn/tineuvox/">
            TiNeuVox (ACM SIGGRAPH Asia 2022)
          </a>
          <a target="_blank" rel="noopener noreferrer" class="navbar-item" href="https://hypernerf.github.io/">
            HyperNeRF (ACM Trans. Graph. 2021)
          </a>
          <a target="_blank" rel="noopener noreferrer" class="navbar-item" href="https://nerfies.github.io/">
            Nerfies (ICCV 2021)
          </a>
          <a target="_blank" rel="noopener noreferrer" class="navbar-item" href="https://www.cs.cornell.edu/~zl548/NSFF/">
            NSFF (CVPR 2021)
          </a>
        </div>
      </div>
    </div>

  </div>-->
</nav>


<section class="hero">
  <div class="hero-body">
    <div class="container is-max-desktop">
      <div class="columns is-centered">
        <div class="column has-text-centered">
          <div class="title is-2 publication-title">
            <h1 style="display: inline; clear: none;">
              <img src="./static/image/globe.png" width="40" height="40">
              <font color="#d91421"><b>U</b></font>-
              <font color="#008000"><b>Know</b></font>-
              <font color="#00BFFF"><b>Diff</b></font>-<b>PAN</b>:
              <font color="#d91421"><b>U</b></font>ncertainty-aware 
              <font color="#008000"><b>Know</b></font>ledge Distillation
              <font color="#00BFFF"><b>Diff</b></font>usion Framework with Details Enhancement for <b>PAN</b>-Sharpening
            </h1>
          </div>  <div class="is-size-5 publication-authors">
            <span class="author-block">
              <a target="_blank" rel="noopener noreferrer" href="https://www.viclab.kaist.ac.kr/">Sungpyo Kim</a><sup> 1</sup>&nbsp;&nbsp;&nbsp;&nbsp;
            </span>
            <span class="author-block">
              <a target="_blank" rel="noopener noreferrer" href="https://www.viclab.kaist.ac.kr/">Jeonghyeok Do</a><sup> 2</sup>&nbsp;&nbsp;&nbsp;&nbsp;
            </span>
            <span class="author-block">
              <a target="_blank" rel="noopener noreferrer" href="https://www.viclab.kaist.ac.kr/">Jaehyeop Lee</a><sup>† 2</sup>
            </span>
            <span class="author-block">
              <a target="_blank" rel="noopener noreferrer" href="https://www.viclab.kaist.ac.kr/">Munchurl Kim</a><sup>† 1</sup>
            </span>
          </div>

          <div class="is-size-5 publication-authors">
            <span class="author-block"> <sup>†</sup>Co-corresponding authors</span>
            <!-- <span class="author-block"><sup>†</sup>Corresponding author</span> -->
          </div>
          <div class="is-size-5 publication-authors">
            <span class="author-block"><sup>1</sup>Korea Advanced Institute of Science and Technology, South Korea</span> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
          </div>
          <div class="is-size-5 publication-authors">
            <span class="author-block"><sup>2</sup>Korea Advanced Institute of Science and Technology, South Korea</span>
          </div>
          <!-- <div class="is-size-4 publication-authors">
            <span class="author-block", style="color:#346dc2">CVPR 2024 Oral</span>
          </div> -->

          <div class="column has-text-centered">
            <div class="publication-links">
              <!-- PDF Link. -->
              <span class="link-block">
                <!-- <a target="_blank" rel="noopener noreferrer" href="https://openaccess.thecvf.com/content/CVPR2024/html/Youk_FMA-Net_Flow-Guided_Dynamic_Filtering_and_Iterative_Feature_Refinement_with_Multi-Attention_CVPR_2024_paper.html" -->
                <a target="_blank" rel="noopener noreferrer" href="https://xxxx.html"
                   class="external-link button is-normal is-rounded is-dark">
                  <span class="icon">
                      <i class="fas fa-file-pdf"></i>
                  </span>
                  <span>Paper</span>
                </a>
              </span>
              <span class="link-block">
                <!-- <a target="_blank" rel="noopener noreferrer" href="http://arxiv.org/abs/2401.03707" -->
                <a target="_blank" rel="noopener noreferrer" href="http://xxxx"
                   class="external-link button is-normal is-rounded is-dark">
                  <span class="icon">
                      <i class="ai ai-arxiv"></i>
                  </span>
                  <span>arXiv</span>
                </a>
              </span>
              <!-- Video Link. -->
              <span class="link-block">
                <!-- <a target="_blank" rel="noopener noreferrer" href="https://www.youtube.com/watch?v=kO7KavOH6vw" -->
                <a target="_blank" rel="noopener noreferrer" href="https://www.youtube.com/xxxxxxxx"
                   class="external-link button is-normal is-rounded is-dark">
                  <span class="icon">
                      <i class="fab fa-youtube"></i>
                  </span>
                  <span>Video</span>
                </a>
              </span>
              <!-- Code Link. -->
              <span class="link-block">
                <!-- <a target="_blank" rel="noopener noreferrer" href="https://github.com/KAIST-VICLab/FMA-Net" -->
                <a target="_blank" rel="noopener noreferrer" href="https://github.com/KAIST-VICLab/xxxxx"
                   class="external-link button is-normal is-rounded is-dark">
                  <span class="icon">
                      <i class="fab fa-github"></i>
                  </span>
                  <span>Code</span>
                  </a>
              </span>
            </div>

          </div>
        </div>
      </div>
    </div>
  </div>
</section>


<!--=================Teasers==========================-->
<section class="hero is-small">
  <div class="hero-body">
    <div class="container is-max-desktop has-text-centered">
      <!-- The expanding image container -->
      <div class="tab_container">
        <!-- Close the image -->
        <!-- <span onclick="this.parentElement.style.display='none'" class="closebtn">&times;</span> -->

        <!-- Expanded image -->
        <div id="juxtapose-embed" data-startingposition="30%" data-animate="true" class="juxtapose"
          style="height: 800px; width: 800px;">
          <div class="jx-slider">
            <div class="jx-handle " style="left: 50%;">
              <div class="jx-arrow jx-left"></div>
              <div class="jx-control">
                <div class="jx-controller" tabindex="0" role="slider" aria-valuenow="50" aria-valuemin="0"
                  aria-valuemax="100"></div>
              </div>
              <div class="jx-arrow jx-right"></div>
            </div>
            <div class="jx-image jx-left " style="width: 50%;"><img src="./src/UDC_1.png" alt="">
              <div class="jx-label" tabindex="0">UDC images</div>
            </div>
            <div class="jx-image jx-right " style="width: 50%;"><img src="./src/AF_1.png" alt="">
              <div class="jx-label" tabindex="0">AlignFormer results</div>
            </div><a href="https://juxtapose.knightlab.com" target="_blank" rel="noopener" class="jx-knightlab">
              <div class="knightlab-logo"></div><span class="juxtapose-name">JuxtaposeJS</span>
            </a>
          </div>
        </div>

        <div>
          <div id="juxtapose-hidden"></div>
        </div>

        <!-- Image text -->
        <div id="imgtext"></div>
      </div>

      <!-- The grid: four columns -->
      <div class="tab_row">
        <div class="tab_column">
          <img src="./static/image/wv3_RR_testimg11_lrms.png" onclick="tab_gallery_click('wv3_RR_testimg11');">
        </div>
        <div class="tab_column">
          <img src="./static/image/wv3_FR_testimg5_lrms.png" onclick="tab_gallery_click('wv3_FR_testimg5');">
        </div>
        <div class="tab_column">
          <img src="./static/image/qb_RR_testimg3_lrms.png" onclick="tab_gallery_click('qb_RR_testimg3');">
        </div>
        <div class="tab_column">
          <img src="./static/image/qb_FR_testimg8_lrms.png" onclick="tab_gallery_click('qb_FR_testimg8');">
        </div>
        <div class="tab_column">
          <img src="./static/image/gf2_RR_testimg19_lrms.png" onclick="tab_gallery_click('gf2_RR_testimg19');">
        </div>
        <div class="tab_column">
          <img src="./static/image/gf2_FR_testimg18_lrms.png" onclick="tab_gallery_click('gf2_FR_testimg18');">
        </div>
      </div>
    </div>
  </div>
</section>

<script>
bulmaCarousel.attach('#results-carousel', {
  slidesToShow: 2,
  loop: true,
  pagination: false,
});
</script>

<section class="section">
  <div class="container is-max-desktop">
    <!-- Abstract. -->
    <div class="columns is-centered has-text-centered">
      <div class="column is-full">
        <h2 class="title is-3">Abstract</h2>
        <div class="content has-text-justified">
          <!-- <p>
            We present a joint learning scheme of video super-resolution and deblurring, called VSRDB, to restore clean high-resolution (HR) videos from blurry low-resolution (LR) ones.
            This joint restoration problem has drawn much less attention compared to single restoration problems.
          </p>
          <p>
            We propose a novel flow-guided dynamic filtering (FGDF) and iterative feature refinement with multi-attention (FRMA), which constitutes our VSRDB framework, denoted as FMA-Net.
            Specifically, our proposed FGDF enables precise estimation of both spatio-temporally-variant degradation and restoration kernels that are aware of motion trajectories through sophisticated motion representation learning.
            Compared to conventional dynamic filtering, the FGDF enables the FMA-Net to effectively handle large motions into the VSRDB.
          </p>
          <p>
            Additionally, the stacked FRMA blocks trained with our novel temporal anchor (TA) loss, which temporally anchors and sharpens features, refine features in a course-to-fine manner through iterative updates.
            Extensive experiments demonstrate the superiority of the proposed FMA-Net over state-of-the-art methods in terms of both quantitative and qualitative quality.
          </p> -->
          <p>
            Conventional methods for PAN-sharpening often struggle to restore fine details due to limitations in leveraging high-frequency information. 
            Moreover, diffusion-based approaches lack sufficient conditioning to fully utilize Panchromatic (PAN) images and low-resolution multispectral (LRMS) inputs effectively. 
            To address these challenges, we propose an uncertainty-aware knowledge distillation diffusion framework with details enhancement for PAN-sharpening, called U-Know-DiffPAN. 
            The U-Know-DiffPAN incorporates uncertainty-aware knowledge distillation for effective transfer of feature details from our teacher model to a student one. 
            The teacher model in our U-Know-DiffPAN captures frequency details through freqeuncy selective attention, facilitating accurate reverse process learning. 
            By conditioning the encoder on compact vector representations of PAN and LRMS and the decoder on Wavelet transforms, we enable rich frequency utilization. 
            So, the high-capacity teacher model distills frequency-rich features into a lightweight student model aided by an uncertainty map. 
            From this, the teacher model can guide the student model to focus on difficult image regions for PAN-sharpening via the usage of the uncertainty map. 
            Extensive experiments on diverse datasets demonstrate the robustness and superior performance of our U-Know-DiffPAN over very recent state-of-the-art PAN-sharpening methods.
          </p>
          <p>

          </p>
        </div>
      </div>
    </div>
    <!--/ Abstract. -->

    <!-- Paper video. -->
    <!-- <div class="columns is-centered has-text-centered">
      <div class="column is-full">
        <h2 class="title is-3">Video</h2>
        <div class="publication-video">
          <iframe src="https://www.youtube.com/embed/G6qqJXztJDM"
                  frameborder="0" allow="autoplay; encrypted-media" allowfullscreen></iframe>
        </div>
      </div>
    </div> -->
    <!--/ Paper video. -->
  </div>
</section>


<section class="section">
  <div class="container is-widescreen">
    <h2 class="title is-3"> Overview of U-Know_Diff_PAN</h2>
    <div class="columns is-centered has-text-centered">
      <div class="column is-full">
        <image src="./static/image/overview.png" 
        class="img-responsive" 
        alt="DDNeRF_Architecture_v21"
        style="width: 75%; height: auto;"><br>
        <div class="content has-text-justified">
          <p>
            The overall pipeline of U-Know-DiffPAN for PAN-sharpening. Our framework operates in two main stages: (i) pre-training the FSA-T within a diffusion process to produce an initial prediction alongside
             an uncertainty map that identifies spatially weak regions; and (ii) training the FSA-S by leveraging this uncertainty map to guide the FSA-S in refining these regions through the KD.
          </p>
        </div>
      </div>
    </div>
  </div>
</section>


<section class="section">
  <div class="container is-widescreen">
    <h2 class="title is-3">Network Architecture</h2>
    <div class="columns is-centered has-text-centered">
      <div class="column is-full">
        <image src="./static/image/FSA_T.png" class="img-responsive" alt="DDNeRF_Architecture_v21"><br>
        <div class="content has-text-justified">
          <p>
            <!-- <b>Top:</b> The overall pipeline of U-Know-DiffPAN for PAN-sharpening. <br>
            <b>Bottom:</b> (a) Architecture of FSA-T block; (b) Architecture of FSA-S block. -->
            Architecture of FSA-T block
          </p>
        </div>
      </div>
    </div>
  </div>
</section>

<section class="section">
  <div class="container is-widescreen">
    <h2 class="title is-3">Quantitative Results</h2>
    <div class="columns is-centered has-text-centered is-multiline">
      <div class="column is-full">
        <image src="./static/image/GF2_table.png" class="img-responsive" alt="DDNeRF_Architecture_v21"><br>
        <div class="content has-text-justified">
          <p>
            Quantitative comparison on GF2 dataset.
            Comparison of different models on the GaoFen-2 (GF2) dataset. Blue indicates the second-best performance, while Red highlights the best-performing model.
          </p>
        </div>
      </div>
      <div class="column is-full">
        <image src="./static/image/WV3_QB_table.png" class="img-responsive" alt="DDNeRF_Architecture_v21"><br>
        <div class="content has-text-justified">
          <p>
            Quantitative comparison on WV3 and QB dataset.
            Comparison of different models on the WorldView-3 (WV3) and QuickBird (QB) datasets. Blue indicates the second-best performance, while Red highlights the best-performing model.
          </p>
        </div>
      </div>
    </div>
  </div>
  </section>

  <section class="section">
    <div class="container is-widescreen">
      <h2 class="title is-3">Qualitative Results</h2>
      <!-- <div class="columns is-centered has-text-centered"> -->
      <div class="columns is-centered has-text-centered is-multiline">
        <div class="column is-full">
          <image src="./static/image/suppl_reduced_gf2.png" class="img-responsive" alt="DDNeRF_Architecture_v21"><br>
          <div class="content has-text-justified">
            <p>
              PAN-sharpening results for the GF2 dataset under reduced resolution (RR) scenarios. The first row depicts the output HRMS
              images, while the second row highlights the Error Map between the output HRMS and the corresponding ground truth images.The Mean
              Absolute Error (MAE) values are presented alongside the Error Map.
            </p>
          </div>
        </div>
        <div class="column is-full">
          <image src="./static/image/suppl_reduced_qb.png" class="img-responsive" alt="DDNeRF_Architecture_v21"><br>
          <div class="content has-text-justified">
            <p>
              PAN-sharpening results for the QB dataset under reduced resolution (RR) scenarios. The first row depicts the output HRMS
              images, while the second row highlights the Error Map between the output HRMS and the corresponding ground truth images. The Mean
              Absolute Error (MAE) values are presented alongside the Error Map.
            </p>
          </div>
        </div>
        <div class="column is-full">
          <image src="./static/image/suppl_reduced_wv3.png" class="img-responsive" alt="DDNeRF_Architecture_v21"><br>
          <div class="content has-text-justified">
            <p>
              PAN-sharpening results for the GF2 dataset under reduced resolution (RR) scenarios. The first row depicts the output HRMS
              images, while the second row highlights the Error Map between the output HRMS and the corresponding ground truth images. The Mean
              Absolute Error (MAE) values are presented alongside the Error Map.
            </p>
          </div>
        </div>
        <div class="column is-full">
          <image src="./static/image/suppl_full_wv3.png" class="img-responsive" alt="DDNeRF_Architecture_v21"><br>
          <div class="content has-text-justified">
            <p>
              PAN-sharpening results for the WV3 dataset under full resolution (FR) scenarios. The first row depicts the output HRMS
              images.
            </p>
          </div>
        </div>
        <div class="column is-full">
          <image src="./static/image/suppl_full_qb.png" class="img-responsive" alt="DDNeRF_Architecture_v21"><br>
          <div class="content has-text-justified">
            <p>
              PAN-sharpening results for the WV3 dataset under full resolution (FR) scenarios. The first row depicts the output HRMS
              images.
            </p>
          </div>
        </div>
        <div class="column is-full">
          <image src="./static/image/suppl_full_gf2.png" class="img-responsive" alt="DDNeRF_Architecture_v21"><br>
          <div class="content has-text-justified">
            <p>
              PAN-sharpening results for the GF2 dataset under full resolution (FR) scenarios. The first row depicts the output HRMS images.
            </p>
          </div>
        </div>
      </div>
    </div>
    </section>

<!-- <section class="section"> -->
  <!-- <div class="container is-widescreen"> -->
    <!-- Concurrent Work. -->
    <!-- <div class="columns is-centered">
      <div class="column is-full-width">
        <h2 class="title is-3">Acknowledgement</h2>

        <div class="content has-text-justified">
          <p>
            This work was supported by the Institute of Information & communications Technology Planning & Evaluation (IITP) grant funded by the Korea government (MSIT): No. 2021-0-00087, Development of high-quality conversion technology for SD/HD low-quality media and No. RS2022-00144444, Deep Learning Based Visual Representational Learning and Rendering of Static and Dynamic Scenes.
          </p>
        </div>
      </div>
    </div> -->
    <!--/ Concurrent Work. -->
  <!-- </div>
</section> -->

<!-- <section class="section" id="BibTeX">
  <div class="container is-widescreen content">
    <h2 class="title">BibTeX</h2>
    <pre><code>@InProceedings{Youk_2024_CVPR,
    author    = {Kim, Sungpyo and Kim, Sungpyo and Kim, Munchurl},
    title     = {U-Know-DiffPAN: An Uncertainty-aware Knowledge Distillation Diffusion Framework with Details Enhancement for PAN-Sharpening},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2024},
    pages     = {44-55}
}</code></pre>
  </div>
</section> -->


<footer class="footer">
  <div class="container">
    <div class="content has-text-centered">
      <a target="_blank" rel="noopener noreferrer" class="icon-link"
         href="">
        <i class="fas fa-file-pdf"></i>
      </a>
      <a target="_blank" rel="noopener noreferrer" class="icon-link" href="https://github.com/KAIST-VICLab" class="external-link" disabled>
        <i class="fab fa-github"></i>
      </a>
    </div>
    <div class="columns is-centered">
      <div class="column is-8">
        <div class="content">
          <p>
            We thank the authors of <a target="_blank" rel="noopener noreferrer" href="https://github.com/nerfies/nerfies.github.io">Nerfies</a>  that kindly open sourced the template of this website.
            Please visit our <a target="_blank" rel="noopener noreferrer" href="https://github.com/KAIST-VICLab">VIC-Lab</a> for more interesting researches
          </p>
        </div>
      </div>
      <a href="https://mapmyvisitors.com/web/1bvps"  title="Visit tracker"><img src="https://mapmyvisitors.com/map.png?d=CCp7XZDLhyWJSzjeYxE-PUcQFV7fsl26aISiaMuAwW8&cl=ffffff" /></a>
    </div>
  </div>
</footer>


<!-- Image Slider Javascripts -->
<!--=================Functions==========================-->
<script src="./static/js/juxtapose.js"></script>

<script>
  var slider;
  let origImages = [
    { "src": "./static/image/wv3_RR_testimg11_lrms.png", "label": "Zoomed LRMS images", },
    { "src": "./static/image/wv3_RR_testimg11_hrms.png", "label": "U-Know-DiffPAN results", }
  ];
  let origOptions = {
    "makeResponsive": true,
    "showLabels": true,
    "mode": "horizontal",
    "showCredits": true,
    "animate": true,
    "startingPosition": "50"
  };

  const juxtaposeSelector = "#juxtapose-embed";
  const transientSelector = "#juxtapose-hidden";


  function tab_gallery_click(name) {
    // Get the expanded image
    let inputImage = {
      label: "Zoomed LRMS images",
    };
    let outputImage = {
      label: "U-Know_DiffPAN results",
    };

    inputImage.src = "./static/image/".concat(name, "_lrms", ".png")
    outputImage.src = "./static/image/".concat(name, "_hrms", ".png")

    let images = [inputImage, outputImage];
    let options = slider.options;
    options.callback = function (obj) {
      var newNode = document.getElementById(obj.selector.substring(1));
      var oldNode = document.getElementById(juxtaposeSelector.substring(1));
      console.log(obj.selector.substring(1));
      console.log(newNode.children[0]);
      oldNode.replaceChild(newNode.children[0], oldNode.children[0]);
      // newNode.removeChild(newNode.children[0]);

    };

    slider = new juxtapose.JXSlider(transientSelector, images, options);
  };

  (function () {
    slider = new juxtapose.JXSlider(
      juxtaposeSelector, origImages, origOptions);
    // document.getElementById("left-button").onclick = replaceLeft;
    // document.getElementById("right-button").onclick = replaceRight;

  })();
  // Get the image text
  var imgText = document.getElementById("imgtext");
  // Use the same src in the expanded image as the image being clicked on from the grid
  // expandImg.src = imgs.src;
  // Use the value of the alt attribute of the clickable image as text inside the expanded image
  imgText.innerHTML = name;
  // Show the container element (hidden with CSS)
  // expandImg.parentElement.style.display = "block";

  $(".flip-card").click(function () {
    console.log("fading in")
    div_back = $(this).children().children()[1]
    div_front = $(this).children().children()[0]
    // console.log($(this).children("div.flip-card-back"))
    console.log(div_back)
    $(div_front).addClass("out");
    $(div_front).removeClass("in");

    $(div_back).addClass("in");
    $(div_back).removeClass("out");

  });

  $(".flip-card").mouseleave(function () {
    console.log("fading in")
    div_back = $(this).children().children()[1]
    div_front = $(this).children().children()[0]
    // console.log($(this).children("div.flip-card-back"))
    console.log(div_back)
    $(div_front).addClass("in");
    $(div_front).removeClass("out");

    $(div_back).addClass("out");
    $(div_back).removeClass("in");

  });
</script>


</body>
</html>