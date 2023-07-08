
# -*- coding: utf-8 -*
import shutil
import numpy as np
import cv2
from PIL import Image
from image_functions import *
from file_functions_2 import *
from segment_class import *
from dominant import *
import kivy
kivy.require("1.9.1")
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.graphics.texture import Texture
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.checkbox import CheckBox
from kivy.uix.gridlayout import GridLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.graphics import Rectangle
from kivy.core.window import Window
from kivy.properties import ObjectProperty
from kivy.lang import Builder
from kivy.uix.screenmanager import Screen, ScreenManager
import numpy as np
import matplotlib.pyplot as plt
import csv




rec_navigator = []
images = []

d_s = [1.000,1.100, 1.115, 1.117, 1.120, 1.1250]  
d_v = [1.000,1.100, 1.250, 1.300, 1.320, 1.340, 1.360]  

def fill_vertical(img):
    thumbnails = []
    textures = []
    img_data = []
    img_bars = []
    if img.brightness >90:
        d_s = [1.000,1.100, 1.115, 1.117, 1.120, 1.1250]  
        d_v = [1.000,1.100, 1.250, 1.300, 1.320, 1.340, 1.360]  
    else:
        d_s = [1.000,1.200, 1.2115, 1.2117, 1.2120, 1.21250]  
        d_v = [1.000,1.200, 1.350, 1.400, 1.45, 1.50, 1.60]  
    thumbnails = load_thumbs(img,d_s,d_v)
    textures = load_textures(thumbnails)
    for j in range(6):
        img_bars.append(get_dominant(thumbnails[j].image.astype('uint8')))
    
    img_bar_textures = load_c_bar_textures(img_bars)

    img_data.append('Brightness ' + str(round(img.brightness,2)))
    img_data.append('Contrast ' + str(round(img.contrast,2)))
    img_data.append('Haze ' + str(round(img.haze_factor,2)))
    img_data.append('Hough Info ' + str(img.hough_info))
    img_data.append('Harris Corners ' + str(img.harris_corners))
    img_data.append('Hough Circles ' + str(img.hough_circles))
    img_data.append('Contour Info ' + str(img.contour_info))
    img_data.append('Variance ' + str(img.variance))
    img_data.append('SHV ' + str(round(img.shv,3)))
    img_data.append('Laplacian ' + str(round(img.laplacian,2)))
    
    filename = os.path.basename(img.fname).split('/')[-1]
    #print(filename)
    r_2 = filename[0][0] 
    img_data.append('Good/Bad: ' + r_2)


    img_data.append('Good/Bad: ' + img.fname[0])
    return(textures,img_bar_textures,img_data)
    #self.image_count = self.image_count + 1



class Menu(Screen):
    # Fecha a tela (menu lateral) voltando para a tela principal
    def fechar_menu(self):
        self.manager.current = 'tela_principal'
    def hor_lo(self):
        self.manager.current = 'hor_lo'
    def fc(self):
        self.manager.current = 'f_chooser' 

    def fill_vertical(textures,img_bar_textures):
        thumbnails = []
        textures = []
        thumbnails = load_thumbs(images[image_current])
        textures = load_textures(thumbnails)
        img_bars = []
        for j in range(6):
            img_bars.append(get_dominant(thumbnails[j].image.astype('uint8')))
        img_bar_textures = load_c_bar_textures(img_bars)
class Horizontal_Image(Screen):
    # Chama outra tela que contém o menu e uma imagem do logotipo

    rectangle_texture_O = ObjectProperty()
    rectangle_texture_1 = ObjectProperty()
    rectangle_texture_2 = ObjectProperty()
    rectangle_texture_3 = ObjectProperty()
    rectangle_texture_4 = ObjectProperty()
    rectangle_texture_5 = ObjectProperty()
    
    butt = ObjectProperty()
    
    gradient_O = ObjectProperty()
    gradient_1 = ObjectProperty()
    gradient_2 = ObjectProperty()
    gradient_3 = ObjectProperty()
    gradient_4 = ObjectProperty()
    gradient_5 = ObjectProperty()

    image_brightness = ObjectProperty()
    image_contrast = ObjectProperty()
    image_haze = ObjectProperty()
    image_hough_info = ObjectProperty()
    image_harris_corners = ObjectProperty()
    image_hough_circles = ObjectProperty()
    image_contour_info = ObjectProperty()
    image_variance = ObjectProperty()
    image_SHV = ObjectProperty()
    image_laplacian = ObjectProperty()
    
    image_orientation = ObjectProperty()

    def clique_iniciar(self):
        self.manager.current = 'menu' 
    def clique_iniciar_2(self):
        self.manager.current = 'tela_principal' 

class Vertical_Image(Screen):

    # Chama outra tela que contém o menu e uma imagem do logotipo
    rectangle_texture_O = ObjectProperty()
    rectangle_texture_1 = ObjectProperty()
    rectangle_texture_2 = ObjectProperty()
    rectangle_texture_3 = ObjectProperty()
    rectangle_texture_4 = ObjectProperty()
    rectangle_texture_5 = ObjectProperty()

    rec_pos_o_x = ObjectProperty()
    rec_pos_1_x = ObjectProperty()
    rec_pos_2_x = ObjectProperty()
    rec_pos_3_x = ObjectProperty()
    rec_pos_4_x = ObjectProperty()
    rec_pos_5_x = ObjectProperty()

    rec_pos_o_y = ObjectProperty()
    rec_pos_1_y = ObjectProperty()
    rec_pos_2_y = ObjectProperty()
    rec_pos_3_y = ObjectProperty()
    rec_pos_4_y = ObjectProperty()
    rec_pos_5_y = ObjectProperty()

    butt = ObjectProperty()
    count = ObjectProperty()    
    i_h = ObjectProperty()
    i_w = ObjectProperty()    
    i_pos_x = ObjectProperty()    
    i_pos_y = ObjectProperty()    
    columns = ObjectProperty()    
    rrows = ObjectProperty()    
    
    gradient_O = ObjectProperty()
    gradient_1 = ObjectProperty()
    gradient_2 = ObjectProperty()
    gradient_3 = ObjectProperty()
    gradient_4 = ObjectProperty()
    gradient_5 = ObjectProperty()

    image_brightness = ObjectProperty()
    image_contrast = ObjectProperty()
    image_haze = ObjectProperty()
    image_hough_info = ObjectProperty()
    image_harris_corners = ObjectProperty()
    image_hough_circles = ObjectProperty()
    image_contour_info = ObjectProperty()
    image_variance = ObjectProperty()
    image_SHV = ObjectProperty()
    image_laplacian = ObjectProperty()
    current_record = ObjectProperty()
    image_stuff = ObjectProperty()
    total_images = ObjectProperty()
    current = ObjectProperty()        

    

    def next_pressed(self,text):
        if images:

            # print('cccccccccccccccccccccccc')
            # print(self.ids.keys())
            # print('gggggggggggggg')

            # print(self.parent.get_screen('tela_principal').ids.keys())
            # print('gggggggggggggg')

            if text ==  'Previous Image' and rec_navigator[0].prev_c_rec > 0:
                rec_navigator[0].c_rec = rec_navigator[0].prev_c_rec - 1
            if text == 'Next Image' and rec_navigator[0].prev_c_rec < len(images) - 1:
                rec_navigator[0].c_rec = rec_navigator[0].prev_c_rec + 1
            if text == 'First Image' or rec_navigator[0].initialize == True:
                rec_navigator[0].c_rec = 0
            if text == 'Last Image':
                rec_navigator[0].c_rec = len(images) - 1
            rec_navigator[0].initialize = False
            rec_navigator[0].first = False
            rec_navigator[0].last = False
            rec_navigator[0].prev_c_rec = rec_navigator[0].c_rec
            self.current_image.text = str(rec_navigator[0].c_rec)
            
        else:
            pass

    def populate_info(self):
            (textures, img_bar_textures, img_data) = fill_vertical(images[rec_navigator[0].c_rec])

            if images[rec_navigator[0].c_rec].orientation  == 'Landscape':    
                #print('Display Landscape',images[rec_navigator[0].c_rec].orientation)
                inc = [450, -650]
                self.image_stuff.i_w = 600
                self.image_stuff.i_h = 400
                self.image_stuff.size = 1300, 1377

                self.image_stuff.x_o = 10
                self.image_stuff.y_o = 967

                self.image_stuff.x_1 = 660
                self.image_stuff.y_1 = 967
 
                self.image_stuff.x_2 = 10
                self.image_stuff.y_2 = 557

                self.image_stuff.x_3 = 660
                self.image_stuff.y_3 = 557
 
                self.image_stuff.x_4 = 10
                self.image_stuff.y_4 = 100

                self.image_stuff.x_5 = 660
                self.image_stuff.y_5 = 100
                
                self.rectangle_texture_O = textures[0]
                self.rectangle_texture_1 = textures[1]
                self.rectangle_texture_2 = textures[2]
                self.rectangle_texture_3 = textures[3]
                self.rectangle_texture_4 = textures[4]
                self.rectangle_texture_5 = textures[5]



                self.gradient_O = img_bar_textures[0]
                self.gradient_1 = img_bar_textures[1]
                self.gradient_2 = img_bar_textures[2]
                self.gradient_3 = img_bar_textures[3]
                self.gradient_4 = img_bar_textures[4]
                self.gradient_5 = img_bar_textures[5]
  



                
            if images[rec_navigator[0].c_rec].orientation  == 'Portrait':    
                
                #print('Display Portrait',images[rec_navigator[0].c_rec].orientation)
                
                inc = (450,-765)

                self.image_stuff.i_w = 400
                self.image_stuff.i_h = 600
                self.image_stuff.size = 1300, 1377
                
                self.image_stuff.x_o = 10
                self.image_stuff.y_o = 765

                self.image_stuff.x_1 = 420
                self.image_stuff.y_1 = 765
 
                self.image_stuff.x_2 = 830
                self.image_stuff.y_2 = 765

                self.image_stuff.x_3 = 10 
                self.image_stuff.y_3 = 77
 
                self.image_stuff.x_4 = 420
                self.image_stuff.y_4 = 77

                self.image_stuff.x_5 = 830
                self.image_stuff.y_5 = 77

                self.rectangle_texture_O = textures[0]
                self.rectangle_texture_1 = textures[1]
                self.rectangle_texture_2 = textures[2]
                self.rectangle_texture_3 = textures[3]
                self.rectangle_texture_4 = textures[4]
                self.rectangle_texture_5 = textures[5]

                self.gradient_O = img_bar_textures[0]
                self.gradient_1 = img_bar_textures[1]
                self.gradient_2 = img_bar_textures[2]
                self.gradient_3 = img_bar_textures[3]
                self.gradient_4 = img_bar_textures[4]
                self.gradient_5 = img_bar_textures[5]
            else:
                pass



            self.image_brightness.text = img_data[0]
            self.image_contrast.text = img_data[1]
            self.image_haze.text = img_data[2]
            self.image_hough_info.text = img_data[3]
            self.image_harris_corners.text = img_data[4]
            self.image_hough_circles.text = img_data[5]
            self.image_contour_info.text = img_data[6]
            self.image_variance.text = img_data[7]
            self.image_SHV.text = img_data[8]
            self.image_laplacian.text =img_data[9] 
            self.current_image.text = str(rec_navigator[0].c_rec)
            

            print(str(rec_navigator[0].c_rec),img_data[10], img_data[9],img_data[8],img_data[4],img_data[1])
            



            rec_navigator[0].first = False
            rec_navigator[0].last = False
            rec_navigator[0].prev_c_rec = rec_navigator[0].c_rec        


    def mmm(self):
        
        checked = []

        if self.tb_o.state == 'down':
            checked.append('Original')                
        if self.tb_1.state == 'down':
            checked.append('Image_1')                
        if self.tb_2.state == 'down':
            checked.append('Image_2')
        if self.tb_3.state == 'down':
            checked.append('Image_3')
        if self.tb_4.state == 'down':
            checked.append('Image_4')
        if self.tb_5.state == 'down':
            checked.append('Image_5')
        if self.tb_d.state == 'down':
            checked.append('Discard')
        num_imgs = len(checked)
        c_rec = images[rec_navigator[0].c_rec]
        if num_imgs == 2:
            for j,thumbnail_name in enumerate(checked):
                if thumbnail_name =='Original':
                    thumbnail = create_thumbnail((d_s[0],d_v[0],1), c_rec,90 )
                    cv2.imshow('Original Image',thumbnail.image.astype('uint8'))
                if thumbnail_name =='Image_1':
                    thumbnail = create_thumbnail((d_s[1],d_v[1],1), c_rec,90 )
                    cv2.imshow('Modified Image 1',thumbnail.image.astype('uint8'))
                if thumbnail_name=='Image_2':
                    thumbnail = create_thumbnail((d_s[2],d_v[2],2), c_rec,90 )
                    cv2.imshow('Modified Image 2',thumbnail.image.astype('uint8'))
                if thumbnail_name =='Image_3':
                    thumbnail = create_thumbnail((d_s[3],d_v[3],3), c_rec,90 )
                    cv2.imshow('Modified Image 3',thumbnail.image.astype('uint8'))
                if thumbnail_name=='Image_4':
                    thumbnail = create_thumbnail((d_s[4],d_v[4],4), c_rec,90 )
                    cv2.imshow('Modified Image 4',thumbnail.image.astype('uint8'))
                if thumbnail_name=='Image_5':
                    thumbnail = create_thumbnail((d_s[5],d_v[5],5), c_rec,90 )
                    cv2.imshow('Modified Image 5',thumbnail.image.astype('uint8'))
            cv2.waitKey(0)

    def clique_iniciar(self):
        self.manager.current = 'menu' 

    def clique_iniciar_2(self):
        self.manager.current = 'hor_lo' 


class FileChooserScreen(Screen):

    def clique_iniciar(self):
        self.manager.current = 'menu' 

    def selected(self,f_names):
        a = 20

        #print(os.path.dirname(f_names[0]))
        globals()['images'] = load_images(get_files('F:/Year 2023/2023-05-05 Glen Allen vs Deep Run Girls Soccer'), a, 0)
        #globals()['images'] = load_images(get_files('F:/Year 2023/2023-05-18 Midlothian vs James River Girls Soccer'), a, 0)
        #globals()['images'] = load_images(get_files('F:/Year 2022/Test/Test_10'), a, 0)
        #globals()['images'] = load_images(get_files('F:/Year 2023/2023-05-09 St Catherines vs Veritas Girls Soccer'),a, 0)
        #globals()['images'] = load_images(get_files(os.path.dirname(f_names[0])),a, 0)
        globals()['rec_navigator'].append(Record_Navigator(len(images)))

        last_directory = "d:/Image Data Files/"  + os.path.basename(os.path.dirname(images[0].fname)) + '.csv'
        data_out = []
        data_in = []

        # ksize = 50
        # sigma = 2
        # theta = np.pi
        # lamda = 1* np.pi
        # gamma = .5
        # phi = 0

        # plt.imshow(kernel)
        # plt.axis('off')  # Remove the axis labels
        # plt.show()

        for image in images:
            data_out.append([os.path.basename(image.fname), image.orientation, round(image.brightness), round(image.contrast), 
                round(image.haze_factor,2), image.hough_info[0],image.hough_info[1],image.hough_circles, image.harris_corners, image.contour_info[0],
                round(image.laplacian), round(image.shv,2), image.variance, image.exposure, image.fstop,
                image.iso, image.b_w[0], image.b_w[1], image.b_w[2],image.faces,image.eyes,image.bodies,image.focal_length,image.classification])
        # Write the data to the CSV file
        with open(last_directory, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(data_out)
        print("Measurements saved to CSV:", last_directory)

        data_in = []
        with open(last_directory, "r") as file:
            reader = csv.reader(file)
            for row in reader:
                first_two_columns = row[:2]
                last_column = row[-1]
                numeric_values = [float(val) for val in row[2:-1]]
                combined_row = first_two_columns + numeric_values + [last_column]
                data_in.append(combined_row)

        # Display the loaded data
        for row in data_in:
            print(row)

            # g = image.image_gs
            # cv2.imshow(os.path.basename(image.fname) + ' Original', g)
            # start = 1.0
            # stop = 6.0
            # step = 0.5
            # for i in range(int(start * 2), int(stop * 2), int(step * 2)):
            #     j = i / 2
            #     kernel = cv2.getGaborKernel((ksize,ksize), sigma+ j,theta/(j+2), lamda/(j + 2),gamma,phi,ktype = cv2.CV_32F)
            #     fimg = cv2.filter2D(g,cv2.CV_8UC3,kernel)
            #     cv2.imshow(os.path.basename(image.fname) + ' Sigma = ' + str(sigma + j) + ' Theta = ' + str(theta/(j + 2)), fimg)
            # cv2.waitKey()





        plt.style.use('seaborn')
        plt.grid(True)  # Enable grid lines

        g_lap_category = []
        b_lap_category = []
        g_cont_category = []
        b_cont_category = []
        g_counter = []
        b_counter = []
        lap_frequency = {}
        cont_frequency = {}

        sorted_images = sorted(images, key=lambda x: x.laplacian)
        for member in sorted_images:
            if os.path.basename(member.fname)[0] =="G":
                g_lap_category.append(round(member.laplacian))
                g_cont_category.append(member.focal_length)
            else:
                b_lap_category.append(round(member.laplacian))
                b_cont_category.append(member.focal_length)

        # Create a line plot
        plt.scatter(g_cont_category, g_lap_category, color='green')
        plt.scatter(b_cont_category, b_lap_category, color='red')

        # Set labels and title
        plt.xlabel('Focal Length')
        plt.ylabel('Laplacian')
        plt.title('Laplacian & Focal Length')

        # Display the plot
        plt.show()
        cv2.waitKey()


        g_lap_category = []
        b_lap_category = []
        g_cont_category = []
        b_cont_category = []

        for member in sorted_images:
            if os.path.basename(member.fname)[0] =="G":
                g_lap_category.append(round(member.laplacian))
                g_cont_category.append(member.contour_info[0])
            else:
                b_lap_category.append(round(member.laplacian))
                b_cont_category.append(member.contour_info[0])

        # Create a line plot
        plt.scatter(g_cont_category, g_lap_category, color='green')
        plt.scatter(b_cont_category, b_lap_category, color='red')

        # Set labels and title
        plt.xlabel('Contours')
        plt.ylabel('Laplacian')
        plt.title('Laplacian & Contours')

        # Display the plot
        plt.show()
        cv2.waitKey()


        g_cont_category = []
        g_hc_category = []
        b_cont_category = []
        b_hc_category = []

        sorted_images = sorted(images, key=lambda x: x.contour_info[0])
        for member in sorted_images:
            print(member)

        for member in sorted_images:
            if os.path.basename(member.fname)[0] =="G":
                g_cont_category.append(member.contour_info[0])
                g_hc_category.append(member.harris_corners)
            else:
                b_cont_category.append(member.contour_info[0])
                b_hc_category.append(member.harris_corners)

        # Create a line plot
        plt.scatter(g_hc_category, g_cont_category, color='green')
        plt.scatter(b_hc_category, b_cont_category, color='red')

        # Set labels and title
        plt.xlabel('Harris Corners')
        plt.ylabel('Contours')
        plt.title('Contours & Harris Corners')

        # Display the plot
        plt.show()
        cv2.waitKey()



        g_cont_category = []
        g_hc_category = []
        b_cont_category = []
        b_hc_category = []

        sorted_images = sorted(images, key=lambda x: x.contour_info[0])
        print('Contours')
        for member in sorted_images:
            print(member)

        for member in sorted_images:
            if os.path.basename(member.fname)[0] =="G":
                g_cont_category.append(member.contour_info[0])
                g_hc_category.append(member.contrast)
            else:
                b_cont_category.append(member.contour_info[0])
                b_hc_category.append(member.contrast)

        # Create a line plot
        plt.scatter(g_hc_category, g_cont_category, color='green')
        plt.scatter(b_hc_category, b_cont_category, color='red')

        # Set labels and title
        plt.xlabel('Contrast')
        plt.ylabel('Contours')
        plt.title('Contours & Contrast')

        # Display the plot
        plt.show()
        cv2.waitKey()




        g_cont_category = []
        g_hc_category = []
        b_cont_category = []
        b_hc_category = []

        sorted_images = sorted(images, key=lambda x: x.contour_info[0])
        print('Contours')
        for member in sorted_images:
            print(member)

        for member in sorted_images:
            if os.path.basename(member.fname)[0] =="G":
                g_cont_category.append(member.contour_info[0])
                g_hc_category.append(member.harris_corners)
            else:
                b_cont_category.append(member.contour_info[0])
                b_hc_category.append(member.harris_corners)

        # Create a line plot
        plt.scatter(g_hc_category, g_cont_category, color='green')
        plt.scatter(b_hc_category, b_cont_category, color='red')

        # Set labels and title
        plt.xlabel('Harris Corners')
        plt.ylabel('Contours')
        plt.title('Contours & Harris Corners')

        # Display the plot
        plt.grid(True)  # Enable grid lines
        plt.show()
        cv2.waitKey()



        g_cont_category = []
        g_hc_category = []
        b_cont_category = []
        b_hc_category = []

        sorted_images = sorted(images, key=lambda x: x.contour_info[0])
        print('Contours')
        for member in sorted_images:
            print(member)

        for member in sorted_images:
            if os.path.basename(member.fname)[0] =="G":
                g_cont_category.append(member.contour_info[0])
                g_hc_category.append(member.variance)
            else:
                b_cont_category.append(member.contour_info[0])
                b_hc_category.append(member.variance)

        # Create a line plot
        plt.scatter(g_hc_category, g_cont_category, color='green')
        plt.scatter(b_hc_category, b_cont_category, color='red')

        # Set labels and title
        plt.xlabel('Contours')
        plt.ylabel('Variance')
        plt.title('Contours & Variance')

        # Display the plot
        plt.show()
        cv2.waitKey()




        g_cont_category = []
        g_hc_category = []
        b_cont_category = []
        b_hc_category = []

        sorted_images = sorted(images, key=lambda x: x.contour_info[0])
        print('Contours')
        for member in sorted_images:
            print(member)

        for member in sorted_images:
            if os.path.basename(member.fname)[0] =="G":
                g_cont_category.append(member.contour_info[0])
                g_hc_category.append(member.hough_info[0])
            else:
                b_cont_category.append(member.contour_info[0])
                b_hc_category.append(member.hough_info[0])

        # Create a line plot
        plt.scatter(g_hc_category, g_cont_category, color='green')
        plt.scatter(b_hc_category, b_cont_category, color='red')

        # Set labels and title
        plt.xlabel('Contours')
        plt.ylabel('Hough Lines')
        plt.title('Contours & Hough Lines')

        # Display the plot
        plt.show()
        cv2.waitKey()


        g_cont_category = []
        b_cont_category = []
        g_shv_category = []
        b_shv_category = []

        sorted_images = sorted(images, key=lambda x: x.shv)
        print('SHV')
        for member in sorted_images:
            print(member)

        for member in sorted_images:
            if os.path.basename(member.fname)[0] =="G":
                g_cont_category.append(member.contour_info[0])
                g_shv_category.append(round(member.shv,2))
            else:
                b_cont_category.append(member.contour_info[0])
                b_shv_category.append(round(member.shv,2))

        # Create a line plot
        plt.scatter(g_shv_category, g_cont_category, color='green')
        plt.scatter(b_shv_category, b_cont_category, color='red')

        # Set labels and title
        plt.xlabel('SHV')
        plt.ylabel('Contours')
        plt.title('Contours & SHV')

        # Display the plot
        plt.show()
        cv2.waitKey()

        g_bright_category = []
        b_bright_category = []
        g_cont_category = []
        b_cont_category = []

        sorted_images = sorted(images, key=lambda x: x.brightness)
        print('Contours')
        for member in sorted_images:
            print(member)

        for member in sorted_images:
            if os.path.basename(member.fname)[0] =="G":
                g_bright_category.append(member.brightness)
                g_cont_category.append(member.contour_info[0])
            else:
                b_bright_category.append(member.brightness)
                b_cont_category.append(member.contour_info[0])

        # Create a line plot
        plt.scatter(g_cont_category, g_bright_category, color='green')
        plt.scatter(b_cont_category, b_bright_category, color='red')

        # Set labels and title
        plt.xlabel('Contours')
        plt.ylabel('Brightness')
        plt.title('Brightness & Contours')

        # Display the plot
        plt.show()
        cv2.waitKey()




        g_cont_category = []
        b_cont_category = []
        g_shv_category = []
        b_shv_category = []



        sorted_images = sorted(images, key=lambda x: x.shv)
        print('SHV')
        for member in sorted_images:
            print(member)

        for member in sorted_images:
            if os.path.basename(member.fname)[0] =="G":
                g_cont_category.append(member.contour_info[0])
                g_shv_category.append(round(member.shv,2))
            else:
                b_cont_category.append(member.contour_info[0])
                b_shv_category.append(round(member.shv,2))

        # Create a line plot
        plt.scatter(g_shv_category, g_cont_category, color='green')
        plt.scatter(b_shv_category, b_cont_category, color='red')

        # Set labels and title
        plt.xlabel('SHV')
        plt.ylabel('Contours')
        plt.title('Contours & SHV')

        # Display the plot
        plt.show()
        cv2.waitKey()


        g_category = []
        b_category = []
        g_counter = []
        b_counter = []


        sorted_images = sorted(images, key=lambda x: x.harris_corners)
        print('Harris Corners')
        for member in sorted_images:
            print(member)
 
        for member in sorted_images:
            if os.path.basename(member.fname)[0] =="G":
                g_category.append(member.harris_corners)
                g_counter.append(member.image_id)
            else:
                b_category.append(member.harris_corners)
                b_counter.append(member.image_id)

        # Create a scatter plot
        plt.scatter(g_counter, g_category, color='green')
        plt.scatter(b_counter, b_category, color='red')

        # Set labels and title
        plt.xlabel('Series')
        plt.ylabel('Harris Corners')
        plt.title('Harris Corners in a Series')

        # Display the plot
        plt.show()
        cv2.waitKey()
 

        self.manager.current = 'tela_principal'



class MyScreenManager(ScreenManager):
    pass
class DisjuntoreseCia(App):
    def build(self):
        # Retornar com função Builder.load_string evita erros em acentuação.
        return Builder.load_string(open('nomealeatorio.kv', encoding='UTF-8').read())
        # return Builder.load_string(open('helloworld_2.kv', encoding='UTF-8').read())

DC = DisjuntoreseCia()
DC.run()        
