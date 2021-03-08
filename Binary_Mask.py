import os

import rasterio
from rasterio.plot import reshape_as_image
import rasterio.mask
from rasterio.features import rasterize

import pandas as pd
import geopandas as gpd
from shapely.geometry import mapping, Point, Polygon
from shapely.ops import cascaded_union

import numpy as np
import cv2
import matplotlib.pyplot as plt

def generate_mask(raster_path, shape_path, output_path, file_name):
    
    """Os CRS devem ser iguais para gerar a máscara binária
    
    raster_path = local onde a imagem .tif esta localizada;

    shape_path = local onde o Shapefile ou GeoJson está localizado.

    output_path = local onde será salva a máscara binária gerada.

    file_name = nome do aquivo que será gerado.
    
    """
    
    #Carregar o Raster
    
    with rasterio.open(raster_path, "r") as src:
        raster_img = src.read()
        raster_meta = src.meta
    
    #Carregar o shapefile ou GeoJson
    train_df = gpd.read_file(shape_path)
    
    #Verificar se o CRS é o mesmo
    if train_df.crs != src.crs:
        print(" crs do Raster : {} e Crs do Vetor : {}.\n Converta para o mesmo Sistema de Coordenadas de Referência!".format(src.crs,train_df.crs))
        
        
    #Função para gerar a máscara
    def poly_from_utm(polygon, transform):
        poly_pts = []

        poly = cascaded_union(polygon)
        for i in np.array(poly.exterior.coords):

            poly_pts.append(~transform * tuple(i))

        new_poly = Polygon(poly_pts)
        return new_poly
    
    
    poly_shp = []
    im_size = (src.meta['height'], src.meta['width'])
    for num, row in train_df.iterrows():
        if row['geometry'].geom_type == 'Polygon':
            poly = poly_from_utm(row['geometry'], src.meta['transform'])
            poly_shp.append(poly)
        else:
            for p in row['geometry']:
                poly = poly_from_utm(p, src.meta['transform'])
                poly_shp.append(poly)

    mask = rasterize(shapes=poly_shp,
                     out_shape=im_size)
    
    #Salvar
    mask = mask.astype("uint16")
    
    bin_mask_meta = src.meta.copy()
    bin_mask_meta.update({'count': 1})
    os.chdir(output_path)
    with rasterio.open(file_name, 'w', **bin_mask_meta) as dst:
        dst.write(mask * 255, 1)


raster_path = '/home/bernard/Documents/IME/Iniciação Científica/Data/Imagem final mosaicada/Apui_mosaicada_3246.tif'
shape_path = '/home/bernard/Documents/IME/Iniciação Científica/Data/Shape_Queimadas/shape_id_1.geojson'
output_path = '/home/bernard/Documents/IME/Iniciação Científica/Data/Binary Masks'
file_name = 'binary_mask_id_1.tif'



generate_mask(raster_path,shape_path,output_path,file_name)