void First_GetPDFEachBin(string filename){
    auto file = new TFile(Form("%s", filename.c_str()), "READ");
    auto NpVSRef3 = (TH2D*)file->Get("hist");

    auto Ref3 = NpVSRef3->ProjectionX();
    int end_point = 0;
    for(int i=2; i<=Ref3->GetNbinsX(); i++){
        if(Ref3->GetBinContent(i) < 30){  // Check if Ref3 bin has >=30 events
            end_point = i;
            break;
        }
    }
    cout << "end_point: " << end_point << endl;
    auto pro_ref3 = (TH1D*)NpVSRef3->ProjectionY(Form("pro_ref3"), 1, end_point);
    
    pro_ref3->Scale(1.0/pro_ref3->GetEntries());
    ofstream ofs("ProtonPDF.txt");
    for(int i=1; i<101; i++){
        ofs << pro_ref3->GetBinCenter(i) << " " << pro_ref3->GetBinContent(i) << " " << pro_ref3->GetBinError(i) << endl;
    }
    ofs.close();


    for(int i=1; i<Ref3->GetNbinsX()+1; i++){
        auto h1 = NpVSRef3->ProjectionY(Form("h2_%d", i), i, i);
        if (h1->Integral() < 30) continue;
        bool isLargeError = false;     
        ofstream ofs(Form("EachBinDistributionFromRef3/Proton_Bin%d.txt", i));
        ofs << fixed << setprecision(4);  //
        for(int j=1; j<=h1->GetNbinsX(); j++){
            double content = h1->GetBinContent(j);
            int entry = h1->GetEntries();
            double error = h1->GetBinError(j);
            ofs << h1->GetBinCenter(j) << " " << content << " " << error << " " << entry << endl;
        }
        ofs.close();
    }
}