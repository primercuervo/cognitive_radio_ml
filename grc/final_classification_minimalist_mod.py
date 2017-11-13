#!/usr/bin/env python2
# -*- coding: utf-8 -*-
##################################################
# GNU Radio Python Flow Graph
# Title: Final Classification Minimalist
# Generated: Sat Nov 11 22:30:34 2017
##################################################

from gnuradio import blocks
from gnuradio import eng_notation
from gnuradio import fft
from gnuradio import gr
from gnuradio.eng_option import eng_option
from gnuradio.fft import window
from gnuradio.filter import firdes
from optparse import OptionParser
import classifier
import fbmc
import numpy as np


class final_classification_minimalist(gr.top_block):

    def __init__(self, SNR, SCN, DC):
        gr.top_block.__init__(self, "Final Classification Minimalist")

        ##################################################
        # Variables
        ##################################################
        self.subchan_map = subchan_map = np.concatenate(([0,]*6, [1,]*116, [0,]*6))
        self.samp_rate = samp_rate = int(1e7)
        self.packetlen_base = packetlen_base = 256 * 8
        self.cfg = cfg = fbmc.fbmc_config(channel_map=(subchan_map), num_payload_bits=packetlen_base, num_overlap_sym=4, samp_rate=int(samp_rate)/4)
        self.taps = taps = np.array([0.003524782368913293, 0.002520402194932103, -3.532667373554307e-17, -0.0025783423334360123, -0.0036887258756905794, -0.0026390093844383955, -3.7046301165340785e-18, 0.0027025998570024967, 0.003868663916364312, 0.0027693307492882013, -9.712380039780164e-18, -0.0028394402470439672, -0.004067056812345982, -0.0029131921473890543, 2.454170927498071e-17, 0.0029908770229667425, 0.004286897834390402, 0.0030728192068636417, -9.712380039780164e-18, -0.0031593774911016226, -0.004531863145530224, -0.0032509532757103443, -6.861573196148834e-18, 0.0033479968551546335, 0.004806521814316511, 0.003451012307778001, -9.712380039780164e-18, -0.0035605679731816053, -0.005116619635373354, -0.0036773078609257936, 2.849619674016194e-17, 0.003801962360739708, 0.005469490308314562, 0.003935364540666342, -9.712380039780164e-18, -0.004078468773514032, -0.0058746375143527985, -0.004232373554259539, -1.196125168755972e-17, 0.004398348741233349, 0.006344608962535858, 0.004577873274683952, -9.712380039780164e-18, -0.004772676154971123, -0.0068963137455284595, -0.004984795115888119, 3.532667373554307e-17, 0.005216646008193493, 0.00755310570821166, 0.005471116863191128, -9.712380039780164e-18, -0.0057516866363584995, -0.00834816973656416, -0.00606258912011981, 9.712380039780164e-18, 0.006409022491425276, 0.009330307133495808, 0.006797448266297579, -9.712380039780164e-18, -0.007235993165522814, -0.010574348270893097, -0.007735027000308037, 9.712380039780164e-18, 0.008307991549372673, 0.012201170437037945, 0.008972631767392159, -9.712380039780164e-18, -0.00975286029279232, -0.014419564977288246, -0.010681704618036747, 9.712380039780164e-18, 0.011806094087660313, 0.017623912543058395, 0.013195047155022621, -9.712380039780164e-18, -0.01495438627898693, -0.022659316658973694, -0.017255060374736786, 9.712380039780164e-18, 0.020392345264554024, 0.03172304481267929, 0.02492397651076317, -9.712380039780164e-18, -0.03204511106014252, -0.052871737629175186, -0.04486315697431564, 9.712380039780164e-18, 0.0747719332575798, 0.15861520171165466, 0.22431577742099762, 0.24915219843387604, 0.22431577742099762, 0.15861520171165466, 0.0747719332575798, 9.712380039780164e-18, -0.04486315697431564, -0.052871737629175186, -0.03204511106014252, -9.712380039780164e-18, 0.02492397651076317, 0.03172304481267929, 0.020392345264554024, 9.712380039780164e-18, -0.017255060374736786, -0.022659316658973694, -0.01495438627898693, -9.712380039780164e-18, 0.013195047155022621, 0.017623912543058395, 0.011806094087660313, 9.712380039780164e-18, -0.010681704618036747, -0.014419564977288246, -0.00975286029279232, -9.712380039780164e-18, 0.008972631767392159, 0.012201170437037945, 0.008307991549372673, 9.712380039780164e-18, -0.007735027000308037, -0.010574348270893097, -0.007235993165522814, -9.712380039780164e-18, 0.006797448266297579, 0.009330307133495808, 0.006409022491425276, 9.712380039780164e-18, -0.00606258912011981, -0.00834816973656416, -0.0057516866363584995, -9.712380039780164e-18, 0.005471116863191128, 0.00755310570821166, 0.005216646008193493, 3.532667373554307e-17, -0.004984795115888119, -0.0068963137455284595, -0.004772676154971123, -9.712380039780164e-18, 0.004577873274683952, 0.006344608962535858, 0.004398348741233349, -1.196125168755972e-17, -0.004232373554259539, -0.0058746375143527985, -0.004078468773514032, -9.712380039780164e-18, 0.003935364540666342, 0.005469490308314562, 0.003801962360739708, 2.849619674016194e-17, -0.0036773078609257936, -0.005116619635373354, -0.0035605679731816053, -9.712380039780164e-18, 0.003451012307778001, 0.004806521814316511, 0.0033479968551546335, -6.861573196148834e-18, -0.0032509532757103443, -0.004531863145530224, -0.0031593774911016226, -9.712380039780164e-18, 0.0030728192068636417, 0.004286897834390402, 0.0029908770229667425, 2.454170927498071e-17, -0.0029131921473890543, -0.004067056812345982, -0.0028394402470439672, -9.712380039780164e-18, 0.0027693307492882013, 0.003868663916364312, 0.0027025998570024967, -3.7046301165340785e-18, -0.0026390093844383955, -0.0036887258756905794, -0.0025783423334360123, -3.532667373554307e-17, 0.002520402194932103])*2
        self.phydyas_taps_time = phydyas_taps_time = np.array(cfg.phydyas_impulse_taps(cfg.num_total_subcarriers(), cfg.num_overlap_sym()))
        self.nguard_bins = nguard_bins = 8
        self.nchan = nchan = 4
        self.sync = sync = fbmc.sync_config(taps=(phydyas_taps_time[1:]/np.sqrt(phydyas_taps_time.dot(phydyas_taps_time))), N=cfg.num_total_subcarriers(), overlap=4, L=cfg.num_total_subcarriers()-1, pilot_A=1.0, pilot_timestep=4, pilot_carriers=(range(8, 118, 12) + [119]), subbands=nchan, bits=packetlen_base, pos=4, u=1, q=4, A=1.0 , fft_len=2**13, guard=nguard_bins)
        self.su_frame_len_low_rate = su_frame_len_low_rate = sync.get_frame_samps(True)
        self.threshold_delta = threshold_delta = 4
        self.su_frame_len = su_frame_len = (su_frame_len_low_rate + len(taps))* 4
        self.pu_frame_len = pu_frame_len = 80*5*9
        self.nfft = nfft = 256

        ##################################################
        # Blocks
        ##################################################

        self.fft_vxx_0 = fft.fft_vcc(nfft, True, (window.blackmanharris(nfft)), True, 1)
        self.classifier_frame_detection_f_0 = classifier.frame_detection_f(su_frame_len/nfft - 9, 5, 5)
        self.classifier_feature_extraction_f_0 = classifier.feature_extraction_f(samp_rate / nfft, 50, pu_frame_len / nfft, 5)
        self.classifier_energy_detection_vcf_0 = classifier.energy_detection_vcf(nfft, 1000, threshold_delta)
        self.blocks_throttle_1 = blocks.throttle(gr.sizeof_gr_complex*1, samp_rate,True)
        self.blocks_stream_to_vector_0 = blocks.stream_to_vector(gr.sizeof_gr_complex*1, nfft)
        self.blocks_null_sink_1 = blocks.null_sink(gr.sizeof_float*1)
        self.blocks_head_1 = blocks.head(gr.sizeof_float*1, 4500)
        self.blocks_file_source_0 = blocks.file_source(gr.sizeof_gr_complex*1, '/home/cuervo/thesis/data/final_pu/{}/scn_{}_snr_{}.dat'.format(DC, SCN, SNR), False)
        self.blocks_file_sink_0_0_3 = blocks.file_sink(gr.sizeof_float*1, 'test/interframe_time_ch_3_scn_{}_snr_{}_{}.dat'.format(SCN, SNR, DC), False)
        self.blocks_file_sink_0_0_3.set_unbuffered(False)
        self.blocks_file_sink_0_0_2 = blocks.file_sink(gr.sizeof_float*1, 'test/interframe_time_ch_4_scn_{}_snr_{}_{}.dat'.format(SCN, SNR, DC), False)
        self.blocks_file_sink_0_0_2.set_unbuffered(False)
        self.blocks_file_sink_0_0_1 = blocks.file_sink(gr.sizeof_float*1, 'test/packet_rate_scn_{}_snr_{}_{}.dat'.format(SCN, SNR, DC), False)
        self.blocks_file_sink_0_0_1.set_unbuffered(False)
        self.blocks_file_sink_0_0_0 = blocks.file_sink(gr.sizeof_float*1, 'test/variance_scn_{}_snr_{}_{}.dat'.format(SCN, SNR, DC), False)
        self.blocks_file_sink_0_0_0.set_unbuffered(False)
        self.blocks_file_sink_0_0 = blocks.file_sink(gr.sizeof_float*1, 'test/interframe_time_ch_2_scn_{}_snr_{}_{}.dat'.format(SCN, SNR, DC), False)
        self.blocks_file_sink_0_0.set_unbuffered(False)
        self.blocks_file_sink_0 = blocks.file_sink(gr.sizeof_float*1, 'test/interframe_time_ch_1_scn_{}_snr_{}_{}.dat'.format(SCN, SNR, DC), False)
        self.blocks_file_sink_0.set_unbuffered(False)

        ##################################################
        # Connections
        ##################################################
        self.connect((self.blocks_file_source_0, 0), (self.blocks_throttle_1, 0))
        self.connect((self.blocks_head_1, 0), (self.blocks_file_sink_0, 0))
        self.connect((self.blocks_stream_to_vector_0, 0), (self.fft_vxx_0, 0))
        self.connect((self.blocks_throttle_1, 0), (self.blocks_stream_to_vector_0, 0))
        self.connect((self.classifier_energy_detection_vcf_0, 0), (self.blocks_null_sink_1, 0))
        self.connect((self.classifier_energy_detection_vcf_0, 1), (self.blocks_null_sink_1, 1))
        self.connect((self.classifier_energy_detection_vcf_0, 2), (self.blocks_null_sink_1, 2))
        self.connect((self.classifier_energy_detection_vcf_0, 3), (self.blocks_null_sink_1, 3))
        self.connect((self.classifier_energy_detection_vcf_0, 4), (self.classifier_frame_detection_f_0, 0))
        self.connect((self.classifier_energy_detection_vcf_0, 7), (self.classifier_frame_detection_f_0, 3))
        self.connect((self.classifier_energy_detection_vcf_0, 5), (self.classifier_frame_detection_f_0, 1))
        self.connect((self.classifier_energy_detection_vcf_0, 6), (self.classifier_frame_detection_f_0, 2))
        self.connect((self.classifier_feature_extraction_f_0, 1), (self.blocks_file_sink_0_0, 0))
        self.connect((self.classifier_feature_extraction_f_0, 5), (self.blocks_file_sink_0_0_0, 0))
        self.connect((self.classifier_feature_extraction_f_0, 4), (self.blocks_file_sink_0_0_1, 0))
        self.connect((self.classifier_feature_extraction_f_0, 3), (self.blocks_file_sink_0_0_2, 0))
        self.connect((self.classifier_feature_extraction_f_0, 2), (self.blocks_file_sink_0_0_3, 0))
        self.connect((self.classifier_feature_extraction_f_0, 0), (self.blocks_head_1, 0))
        self.connect((self.classifier_frame_detection_f_0, 0), (self.classifier_feature_extraction_f_0, 0))
        self.connect((self.classifier_frame_detection_f_0, 1), (self.classifier_feature_extraction_f_0, 1))
        self.connect((self.fft_vxx_0, 0), (self.classifier_energy_detection_vcf_0, 0))

        ##################################################
        # Run the flowgraph
        ##################################################
        self.run()

    def get_subchan_map(self):
        return self.subchan_map

    def set_subchan_map(self, subchan_map):
        self.subchan_map = subchan_map

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.blocks_throttle_1.set_sample_rate(self.samp_rate)

    def get_packetlen_base(self):
        return self.packetlen_base

    def set_packetlen_base(self, packetlen_base):
        self.packetlen_base = packetlen_base

    def get_cfg(self):
        return self.cfg

    def set_cfg(self, cfg):
        self.cfg = cfg

    def get_taps(self):
        return self.taps

    def set_taps(self, taps):
        self.taps = taps
        self.set_su_frame_len((self.su_frame_len_low_rate + len(self.taps))* 4)

    def get_phydyas_taps_time(self):
        return self.phydyas_taps_time

    def set_phydyas_taps_time(self, phydyas_taps_time):
        self.phydyas_taps_time = phydyas_taps_time

    def get_nguard_bins(self):
        return self.nguard_bins

    def set_nguard_bins(self, nguard_bins):
        self.nguard_bins = nguard_bins

    def get_nchan(self):
        return self.nchan

    def set_nchan(self, nchan):
        self.nchan = nchan

    def get_sync(self):
        return self.sync

    def set_sync(self, sync):
        self.sync = sync

    def get_su_frame_len_low_rate(self):
        return self.su_frame_len_low_rate

    def set_su_frame_len_low_rate(self, su_frame_len_low_rate):
        self.su_frame_len_low_rate = su_frame_len_low_rate
        self.set_su_frame_len((self.su_frame_len_low_rate + len(self.taps))* 4)

    def get_threshold_delta(self):
        return self.threshold_delta

    def set_threshold_delta(self, threshold_delta):
        self.threshold_delta = threshold_delta
        self.classifier_energy_detection_vcf_0.set_threshold_delta_db(self.threshold_delta)

    def get_su_frame_len(self):
        return self.su_frame_len

    def set_su_frame_len(self, su_frame_len):
        self.su_frame_len = su_frame_len

    def get_pu_frame_len(self):
        return self.pu_frame_len

    def set_pu_frame_len(self, pu_frame_len):
        self.pu_frame_len = pu_frame_len

    def get_nfft(self):
        return self.nfft

    def set_nfft(self, nfft):
        self.nfft = nfft


##################################################
# Settings
##################################################

SNR = ['-5', '-2_5', '0', '2_5', '5', '10', '15']
SCN = [x for x in range(10)]
DC = ['no_dc', 'with_dc']


def main(top_block_cls=final_classification_minimalist, options=None):
    for snr in SNR:
        for scn in SCN:
            for dc in DC:
                flowgraph = final_classification_minimalist(snr, scn, dc)
                print(("Features engineering for Scenario {} with SNR {} dB"
                      " and {} offset DONE!").format(scn, snr, dc))

if __name__ == '__main__':
    main()
