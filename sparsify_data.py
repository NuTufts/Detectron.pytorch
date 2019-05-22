import os,sys
import ROOT as rt
from larcv import larcv
from ROOT import std



def sparsify(inputfile, outputfile,
             adc_producer="adc", cluster_producer="masks"):


    io = larcv.IOManager(larcv.IOManager.kREAD,"")
    io.add_in_file(inputfile)
    # io.specify_data_read(larcv.kProductImage2D,adc_producer)
    # io.specify_data_read(larcv.kProductClusterMask,cluster_producer)
    io.initialize()

    out = larcv.IOManager(larcv.IOManager.kWRITE,"")
    out.set_out_file(outputfile)
    out.initialize()

    # flowdef_list = [(2,0,1,4,5)] # (src,tar1,tar2,flow-index-1,flow-index-2)
    nentries = io.get_n_entries()
    nentries = 10
    for ientry in range(nentries):
        io.read_entry(ientry)

        ev_adc  = io.get_data("image2d","adc")
        ev_cluster_in = io.get_data("clustermask","masks")
        adc_v  = ev_adc.image2d_array()
        cluster_vv = ev_cluster_in.clustermask_array()

        nimgs = 3

        threshold_v = std.vector("float")(1,5.0)
        cuton_pixel_v = std.vector("int")(nimgs,1)

        producername_adc = "adc_sparse"
        producername_masks = "masks"
        # if nflows==1:
        #     producername += "_"+flowdirs[0]
        ev_sparse  = out.get_data("sparseimage",producername_adc)
        ev_cluster_out = out.get_data("clustermask", producername_masks)

        for p in range(len(adc_v)):
            # img_1v = std.vector("larcv::Image2D")(1)
            # img_1v[0] = adc_v[p]
            # cuton_1v = std.vector("int")(1,cuton_pixel_v[p])
            # threshold_1v = std.vector("float")(1,threshold_v[p])

            adc_sparse_tensor = larcv.SparseImage(adc_v[p], adc_v[p], threshold_v)
            print("number of sparse floats: ",adc_sparse_tensor.pixellist().size())
            sparse_nd = larcv.as_ndarray(adc_sparse_tensor,larcv.msg.kDEBUG)

            ncols = adc_v.front().meta().cols()
            nrows = adc_v.front().meta().rows()
            maxpixels = ncols*nrows
            occupancy_frac = float(sparse_nd.shape[0])/maxpixels
            print("SparseImage shape: ",sparse_nd.shape," occupancy=",occupancy_frac)

            ev_sparse.Append( adc_sparse_tensor )

        for p in range(len(cluster_vv)):
            print(cluster_vv[p][0].meta.width())
            ev_cluster_out.append(cluster_vv[p])

        out.set_id( io.event_id().run(),
                    io.event_id().subrun(),
                    io.event_id().event() )
        out.save_entry()
        print("Filled Event %d"%(ientry))
        #break

    out.finalize()
    io.finalize()

if __name__ == "__main__":
    """
    run a test example.
    """

    larcv_mctruth     = sys.argv[1]
    output_sparsified = sys.argv[2]

    #sparsify( "../testdata/mcc9mar_bnbcorsika/larcv_mctruth_ee881c25-aeca-4c92-9622-4c21f492db41.root",
    #          "out_sparsified.root" )

    sparsify( larcv_mctruth, output_sparsified )

    #output_sparsified_y2u = output_sparsified.replace(".root","_y2u.root")
    #sparsify( larcv_mctruth, output_sparsified_y2u, flowdirs=['y2u'] )

    #output_sparsified_y2v = output_sparsified.replace(".root","_y2v.root")
    #sparsify( larcv_mctruth, output_sparsified_y2v, flowdirs=['y2v'] )
